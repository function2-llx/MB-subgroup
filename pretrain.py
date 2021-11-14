from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.transforms import LoadImageD, Lambda
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import generate_model
from runner_base import RunnerBase
from utils.args import ArgumentParser, PretrainArgs
from utils.data import MultimodalDataset

class Pretrainer(RunnerBase):
    def __init__(self, args: PretrainArgs):
        super().__init__(args)
        self.args = args

        assert args.model == 'resnet'
        self.model = generate_model(args, pretrain=False, num_seg=3).to(self.args.device)

    def prepare_data(self):
        from utils.data.datasets.brats21 import load_all
        data = load_all(self.args)

        def loader(data):
            data = dict(data)
            data_path = data.pop('data')
            return {**data, **np.load(data_path)}

        train_transforms = RunnerBase.get_train_transforms(self.args)
        return MultimodalDataset(data, [Lambda(loader)] + train_transforms, progress=True, cache_num=300)

    def train(self):
        torch.backends.cudnn.benchmark = True
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        optimizer = Adam(
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
                save_path: Path = output_dir / f'ep{epoch}' / f'state.pth'
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
                loss = loss_fn(outputs.seg, data['seg'].to(self.args.device))
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
