from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import monai
import numpy as np
import torch
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.utils import InterpolateMode
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import TrainingArguments, IntervalStrategy

from models import generate_model
from runner_base import RunnerBase
from utils.args import ArgumentParser, DataTrainingArgs, ModelArgs
from utils.data import MultimodalDataset
from utils.transforms import CreateForegroundMaskD, RandSampleSlicesD

@dataclass
class PretrainArgs(DataTrainingArgs, ModelArgs, TrainingArguments):
    def __post_init__(self):
        self.save_strategy = IntervalStrategy.EPOCH
        super().__post_init__()

class Pretrainer(RunnerBase):
    def __init__(self, args: PretrainArgs):
        super().__init__(args)
        self.args = args

        assert args.model == 'resnet'
        self.model = generate_model(args, pretrain=False, num_seg=3).to(self.args.device)

    @classmethod
    def get_train_transforms(cls, args: PretrainArgs) -> List[monai.transforms.Transform]:
        all_keys = ['img', 'seg', 'fg_mask']
        img_keys = ['img']
        if args.input_fg_mask:
            img_keys.append('fg_mask')

        def loader(data):
            data = dict(data)
            data_path = data.pop('data')
            return {**data, **np.load(data_path)}

        ret: List[monai.transforms.Transform] = [
            monai.transforms.Lambda(loader),
            CreateForegroundMaskD('img', 'fg_mask'),
            monai.transforms.NormalizeIntensityD('img', channel_wise=True, nonzero=args.input_fg_mask),
            monai.transforms.ConcatItemsD(img_keys, 'img'),
            RandSampleSlicesD(all_keys, args.sample_slices),
        ]

        if 'flip' in args.aug:
            ret.extend([
                monai.transforms.RandFlipD(all_keys, prob=0.5, spatial_axis=0),
                monai.transforms.RandFlipD(all_keys, prob=0.5, spatial_axis=1),
                monai.transforms.RandFlipD(all_keys, prob=0.5, spatial_axis=2),
                monai.transforms.RandRotate90D(all_keys, prob=0.5, max_k=1),
            ])
        if 'crop' in args.aug:
            ret.append(monai.transforms.RandSpatialCropD(
                all_keys,
                roi_size=(100, 100, -1),
                random_center=True,
                random_size=True,
            ))
        if 'noise' in args.aug:
            ret.extend([
                monai.transforms.RandScaleIntensityD(args.protocols, factors=0.1, prob=1),
                monai.transforms.RandShiftIntensityD(args.protocols, offsets=0.1, prob=1),
            ])
        if 'blur' in args.aug:
            ret.append(monai.transforms.RandGaussianSmoothD(args.protocols, prob=0.5))

        ret.extend([
            monai.transforms.ResizeD('img', spatial_size=(args.sample_size, args.sample_size, -1)),
            monai.transforms.ResizeD(
                ['seg', 'fg_mask'],
                spatial_size=(args.sample_size, args.sample_size, -1),
                mode=InterpolateMode.NEAREST,
            ),
            monai.transforms.ConcatItemsD(img_keys, 'img'),
            monai.transforms.CastToTypeD('img', np.float32),
            monai.transforms.CastToTypeD('seg', np.int8),
            monai.transforms.ToTensorD(['img', 'seg', 'label'], device=args.device),
        ])

        return ret

    def prepare_data(self):
        from utils.data.datasets.brats21 import load_all
        data = load_all(self.args)

        train_transforms = self.get_train_transforms(self.args)
        return MultimodalDataset(data, train_transforms, progress=True, cache_num=300)

    def train(self):
        output_dir = Path(self.args.output_dir)
        loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            eps=self.args.adam_epsilon,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            amsgrad=True,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=int(self.args.num_train_epochs))
        dataset = self.prepare_data()
        loader = DataLoader(dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True)
        if self.is_world_master():
            writer = SummaryWriter(log_dir=str(Path('runs') / self.args.output_dir))

        start_epoch = 1
        if not self.args.overwrite_output_dir:
            states = None
            for epoch in range(1, int(self.args.num_train_epochs) + 1):
                save_path: Path = output_dir / f'checkpoint-ep{epoch}.pth.tar'
                if save_path.exists():
                    states = torch.load(save_path)
                    start_epoch = epoch + 1
            if states is not None:
                self.model.load_state_dict(states['state_dict'])
                optimizer.load_state_dict(states['optimizer'])
                scheduler.load_state_dict(states['scheduler'])

        for epoch in range(start_epoch, int(self.args.num_train_epochs) + 1):
            epoch_loss = 0
            for data in tqdm(loader, ncols=80, desc=f'epoch{epoch}'):
                outputs = self.model.forward(data['img'].to(self.args.device), permute=True)
                loss = loss_fn(outputs.seg_logit, data['seg'].to(self.args.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            if self.is_world_master():
                writer.add_scalar('loss', epoch_loss / len(loader), epoch)
                # if epoch % self.args.save_strategy == 0:
                save_states = {
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                save_path: Path = output_dir / f'ep{epoch}' / f'state.pth'
                save_path.parent.mkdir(exist_ok=True)
                torch.save(save_states, save_path)

def get_loader(dataset) -> Callable[[Namespace], MultimodalDataset]:
    import utils.data.datasets as datasets
    loader = {
        'brats21': datasets.brats21.load_all,
    }[dataset]
    return loader

def main():
    parser = ArgumentParser([PretrainArgs])
    args, = parser.parse_args_into_dataclasses()
    trainer = Pretrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
