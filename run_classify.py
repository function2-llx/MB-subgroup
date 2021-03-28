import logging
from pathlib import Path
from typing import Optional

import torch
import torch_optimizer as optim
from monai.data import DataLoader
from torch import nn
from torch.nn import CrossEntropyLoss, DataParallel, functional as F
from tqdm import tqdm

from medical_net.model import generate_model
from medical_net.models.resnet import ResNet
from runner_base import RunnerBase
from utils.data import MultimodalDataset, load_folds


def get_args():
    from args import parser, parse_args
    parser.add_argument('--output_root', type=Path, default='output_classify')
    parser.add_argument('--weight_strategy', choices=['invsqrt', 'equal', 'inv'], default='inv')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--targets', choices=['all', 'G3G4'], default='all')
    parser.add_argument('--sample_size',
                        default=448,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--sample_slices',
                        default=24,
                        type=int,
                        help='slices of inputs, temporal size in terms of videos')
    args = parse_args()
    args.model_output_root = args.output_root \
        / f'{args.targets}' \
        / f'{args.aug}_aug' \
        / f'd{args.model_depth}' \
        / 'bs{batch_size},lr{lr},{weight_strategy},{sample_size},{sample_slices}'.format(**args.__dict__)
    args.resnet_shortcut = {
        10: 'B',
        18: 'A',
        34: 'A',
        50: 'B',
    }[args.model_depth]
    args.pretrain_path = Path(f'medical_net/pretrain/resnet_{args.model_depth}_23dataset.pth')
    args.target_names = {
        'all': ['WNT', 'SHH', 'G3', 'G4'],
        'G3G4': ['G3', 'G4'],
    }[args.targets]
    args.rank = 0
    args.target_dict = {name: i for i, name in enumerate(args.target_names)}
    return args

class Runner(RunnerBase):
    def __init__(self, args, folds):
        super().__init__(args, folds)

    def get_grouped_parameters(self, model: nn.Module):
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        no_decay = ['fc']
        grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-5,
             'lr': self.args.lr},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.args.lr},
        ]
        return grouped_parameters

    def run_fold(self, val_id: int):
        output_path = self.args.model_output_root / f'checkpoint-{val_id}.pth.tar'
        if self.args.train:
            logging.info(f'run cross validation on fold {val_id}')
            model = generate_model(self.args, pretrain=True)
            train_set, val_set = self.prepare_fold(val_id)
            logging.info(f'''{torch.cuda.device_count()} GPUs available\n''')
            model = nn.DataParallel(model)
            train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
            loss_fn = CrossEntropyLoss(weight=train_set.get_weight(self.args.weight_strategy).to(self.args.device))
            optimizer = optim.AdaBelief(self.get_grouped_parameters(model))
            best_loss = float('inf')
            patience = 0
            for epoch in range(1, self.args.epochs + 1):
                model.train()
                for data in tqdm(train_loader, desc=f'training: epoch {epoch}', ncols=80):
                    imgs = data['img'].to(self.args.device)
                    labels = data['label'].to(self.args.device)
                    logits = model.forward(type(model.module).classify, imgs)
                    loss = loss_fn(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                val_loss = self.run_eval(model.module, val_set)
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

        model = generate_model(self.args, pretrain=False)
        model.load_state_dict(torch.load(output_path))
        self.run_eval(model, val_set, 'cross-val')

    def run_eval(self, model: ResNet, eval_dataset: MultimodalDataset, test_name: Optional[str] = None) -> float:
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for data in tqdm(DataLoader(eval_dataset, batch_size=1, shuffle=False), ncols=80, desc='evaluating'):
                img = data['img'].to(self.args.device)
                label = data['label'].to(self.args.device)
                logit = model.classify(img)
                if test_name:
                    self.reporters[test_name].append(logit[0], label.item())

                loss = F.cross_entropy(logit, label)
                eval_loss += loss.item()
        return eval_loss


if __name__ == '__main__':
    args = get_args()
    runner = Runner(args, load_folds(args))
    runner.run()
