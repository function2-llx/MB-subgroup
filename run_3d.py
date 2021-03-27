import logging
from pathlib import Path
from typing import Optional

import torch
import torch_optimizer as optim
from monai.data import DataLoader
from torch import nn
from torch.nn import CrossEntropyLoss, DataParallel, functional as F
from tqdm import tqdm

from resnet_3d.utils import get_pretrain_config
from runner_base import RunnerBase
from utils.data_3d import MultimodalDataset, load_folds


def generate_model(pretrained_name: str, n_output: Optional[int] = None) -> nn.Module:
    from resnet_3d import model
    config = get_pretrain_config(pretrained_name)

    model = getattr(model, config['type']).generate_model(
        config['model_depth'],
        n_classes=config['n_pretrain_classes'] if n_output is None else n_output,
        n_input_channels=3,
    )
    return model

def load_pretrained_model(pretrained_name: str, n_output: int):
    from resnet_3d.model import load_pretrained_model, pretrained_root
    model = generate_model(pretrained_name)
    return load_pretrained_model(model, pretrained_root / f'{pretrained_name}.pth', 'resnet', n_output)

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--weight_strategy', choices=['invsqrt', 'equal', 'inv'], default='equal')
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--output_root', type=Path, default='output_3d')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--targets', choices=['all', 'G3G4'], default='all')
    parser.add_argument(
        '--pretrained_name',
        default='r3d18_KM_200ep',
        choices=[
            'r3d18_K_200ep',
            'r3d18_KM_200ep',
            'r3d34_K_200ep',
            'r3d34_KM_200ep',
            'r3d50_KMS_200ep',
            'r2p1d18_K_200ep',
            'r2p1d34_K_200ep',
        ],
        type=str,
        help='Pretrained model name'
    )
    parser.add_argument('--sample_size',
                        default=112,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--sample_slices',
                        default=16,
                        type=int,
                        help='slices of inputs, temporal size in terms of videos')
    parser.add_argument('--aug', choices=['no', 'weak', 'strong'], default='no')
    args = parser.parse_args()
    args.target_names = {
        'all': ['WNT', 'SHH', 'G3', 'G4'],
        'G3G4': ['G3', 'G4'],
    }[args.targets]
    args.target_dict = {name: i for i, name in enumerate(args.target_names)}
    args.model_output_root = args.output_root \
        / f'{args.targets}' \
        / f'{args.aug}_aug' \
        / f'{args.pretrained_name}' \
        / 'bs{batch_size},lr{lr},{weight_strategy},{sample_size},{sample_slices}'.format(**args.__dict__)

    return args

class Runner(RunnerBase):
    def run_fold(self, val_id: int):
        output_path = self.args.model_output_root / f'checkpoint-{val_id}.pth.tar'
        if self.args.train:
            logging.info(f'run cross validation on fold {val_id}')
            model = load_pretrained_model(self.args.pretrained_name, len(self.args.target_names))
            train_set, val_set = self.prepare_fold(val_id)
            if self.args.device == 'cuda' and torch.cuda.device_count() > 1:
                logging.info(f'''using {torch.cuda.device_count()} GPUs\n''')
                model = nn.DataParallel(model)
            model = model.to(self.args.device)
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
            loss_fn = CrossEntropyLoss(weight=train_set.get_weight(self.args.weight_strategy).to(self.args.device))
            optimizer = optim.AdaBelief(self.get_grouped_parameters(model))
            best_loss = float('inf')
            patience = 0
            for epoch in range(1, self.args.epochs + 1):
                model.train()
                for inputs, labels in tqdm(train_loader, desc=f'training: epoch {epoch}', ncols=80):
                    # inputs = inputs.to(self.args.device)
                    labels = labels.to(self.args.device)
                    logits = model.forward(inputs)
                    loss = loss_fn(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                val_loss = self.run_eval(model, val_set)
                logging.info(f'cur loss:  {val_loss}')
                logging.info(f'best loss: {best_loss}')

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(
                        model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                        output_path,
                    )
                    logging.info(f'model updated, saved to {output_path}\n')
                    patience = 0
                else:
                    patience += 1
                    logging.info(f'patience {patience}/{self.args.patience}\n')
                    if patience == self.args.patience:
                        logging.info('run out of patience\n')
                        break
        else:
            val_set = self.prepare_val_fold(val_id)

        model = generate_model(self.args.pretrained_name, len(self.args.target_names)).to(self.args.device)
        model.load_state_dict(torch.load(output_path))
        self.run_eval(model, val_set, 'cross-val')

    def run_eval(self, model: nn.Module, eval_dataset: MultimodalDataset, test_name: Optional[str] = None) -> float:
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for data in tqdm(DataLoader(eval_dataset, batch_size=1, shuffle=False), ncols=80, desc='evaluating'):
                img = data['img'].to(self.args.device)
                label = data['label'].to(self.args.device)
                logit = model.forward(img)
                if test_name:
                    self.reporters[test_name].append(logit[0], label.item())

                loss = F.cross_entropy(logit, label)
                eval_loss += loss.item()
        return eval_loss

if __name__ == '__main__':
    args = get_args()
    runner = Runner(get_args(), load_folds(args))
    runner.run()
