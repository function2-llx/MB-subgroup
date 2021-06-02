from argparse import Namespace
from pathlib import Path
from typing import Callable

import torch
from monai.data import DataLoader
from monai.losses import DiceLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import generate_model, permute_img
from runner_base import RunnerBase
from utils.data import MultimodalDataset

def parse_args():
    from argparse import ArgumentParser
    import utils.args
    import models

    parser = ArgumentParser(parents=[utils.args.parser, models.args.parser])
    parser.add_argument('--output_root', type=Path, default='pretrained')
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--datasets', choices=['brats20'], default=['brats20'])
    args = parser.parse_args()
    args = utils.args.process_args(args)
    args = models.process_args(args)
    args.model = 'resnet'
    args.n_classes = None
    args.model_output_root = args.output_root \
        / '+'.join(args.datasets) \
        / '{aug_list},bs{batch_size},lr{lr},wd{weight_decay},{sample_size}x{sample_slices}'.format(
            aug_list='+'.join(args.aug) if args.aug else 'no',
            **args.__dict__,
        )
    print('output root:', args.model_output_root)
    args.rank = 0

    return args

class Pretrainer(RunnerBase):
    def __init__(self, args):
        super().__init__(args)

        assert args.model == 'resnet'
        self.model = generate_model(args, pretrain=False, num_seg=3).to(self.args.device)

    def prepare_data(self):
        from utils.data.datasets.brats20 import load_all
        data = load_all(self.args)
        return MultimodalDataset(data, self.get_train_transforms(with_seg=True), progress=False)

    def train(self):
        self.args.model_output_root.mkdir(exist_ok=True, parents=True)
        dataset = self.prepare_data()
        loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, amsgrad=True)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        if self.is_world_master():
            writer = SummaryWriter(log_dir=Path('runs') / self.args.model_output_root)

        for epoch in range(1, self.args.epochs + 1):
            epoch_loss = 0
            for data in tqdm(loader, ncols=80, desc=f'epoch{epoch}'):
                outputs = self.model.forward(permute_img(data['img'].to(self.args.device)))
                loss = loss_fn(outputs['seg'], permute_img(data['seg'].to(self.args.device)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if self.is_world_master():
                writer.add_scalar('loss', epoch_loss, epoch)
                if epoch % self.args.save_epoch == 0:
                    save_states = {
                        'state_dict': self.model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'scheduler': scheduler.state_dict()
                    }
                    save_path: Path = self.args.model_output_root / f'ep{epoch}' / f'state.pth'
                    save_path.parent.mkdir(exist_ok=True)
                    torch.save(save_states, save_path)

def get_loader(dataset) -> Callable[[Namespace], MultimodalDataset]:
    import utils.data.datasets as datasets
    loader = {
        'brats20': datasets.brats20.load_all,
    }[dataset]
    return loader

def main():
    trainer = Pretrainer(parse_args())
    trainer.train()

if __name__ == '__main__':
    main()
