import logging
from pathlib import Path
from typing import Optional

import torch
import torch_optimizer as optim
from monai.data import DataLoader
from torch import nn
from torch.nn import MSELoss, DataParallel
from torch.nn import functional as F
from tqdm import tqdm

from medical_net.model import generate_model
from medical_net.models.resnet import ResNet
from finetuner_base import FinetunerBase
from utils.data import load_folds, MultimodalDataset, BalancedSampler


def get_args():
    from args import parser, parse_args
    # parser.add_argument('--weight_strategy', choices=['invsqrt', 'equal', 'inv'], default='equal')
    parser.add_argument('--output_root', type=Path, default='output_siamese')
    parser.add_argument('--sample_size',
                        default=224,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--sample_slices',
                        default=24,
                        type=int,
                        help='slices of inputs, temporal size in terms of videos')
    args = parse_args()
    args.rank = 0
    args.model_output_root = args.output_root \
        / f'{args.targets}' \
        / f'{args.aug}_aug' \
        / f'd{args.model_depth}' \
        / 'bs{batch_size},lr{lr},{sample_size},{sample_slices}'.format(**args.__dict__)
    args.resnet_shortcut = {
        10: 'B',
        18: 'A',
        34: 'A',
        50: 'B',
    }[args.model_depth]
    args.pretrain_path = Path(f'medical_net/pretrain/resnet_{args.model_depth}_23dataset.pth')
    return args

class Runner(FinetunerBase):
    def __init__(self, conf, folds):
        super().__init__(conf, folds)
        self.loss_fn = MSELoss()

    def get_grouped_parameters(self, model: nn.Module):
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        no_decay = ['fc']
        grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-5,
             'lr': self.args.lr},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.args.lr * 5},
        ]
        return grouped_parameters

    def forward(self, features, ref_features, labels, ref_labels, reg=False):
        dis = torch.mean(
            F.mse_loss(
                features[:, None, :].repeat(1, ref_features.shape[0], 1),
                ref_features[None, :, :].repeat(features.shape[0], 1, 1),
                reduction='none',
            ),
            dim=-1,
        )
        pos_mask = torch.eq(
            labels[:, None].repeat(1, ref_labels.shape[0]),
            ref_labels[None, :].repeat(labels.shape[0], 1)
        ).bool()
        loss = (dis[pos_mask].sum() - dis[~pos_mask].sum()) / dis.numel()
        if reg:
            loss += (features ** 2).mean()

        pred = ref_labels[dis.argmin(dim=1)]
        return loss, pred

    def run_fold(self, val_id: int):
        output_path = self.args.model_output_root / f'checkpoint-{val_id}.pth.tar'
        train_set, val_set = self.prepare_fold(val_id)
        if self.args.train:
            logging.info(f'run cross validation on fold {val_id}')
            # model = load_pretrained_model(self.args.pretrained_name)
            model = generate_model(args, pretrain=True)
            model = model.to(self.args.device)
            model = DataParallel(model)
            train_loader = DataLoader(
                train_set,
                batch_size=self.args.batch_size,
                sampler=BalancedSampler(train_set, total=self.args.val_steps),
            )
            optimizer = optim.AdaBelief(self.get_grouped_parameters(model))
            best_loss = float('inf')
            patience = 0
            for epoch in range(1, self.args.epochs + 1):
                model.train()
                for data in tqdm(train_loader, desc=f'training: epoch {epoch}', ncols=80):
                    optimizer.zero_grad()
                    features = model.forward(type(model.module).feature, data['img'].to(self.args.device))
                    loss, _ = self.forward(features, features, data['label'], data['label'], reg=True)
                    loss.backward()
                    optimizer.step()
                val_loss = self.run_eval(model.module, train_set, val_set)
                logging.info(f'cur loss:  {val_loss}')
                logging.info(f'best loss: {best_loss}')

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.module.state_dict(), output_path)
                    logging.info(f'model updated, saved to {output_path}\n')
                    patience = 0
                else:
                    patience += 1
                    logging.info(f'patience {patience}/{self.args.patience}\n')
                    if patience == self.args.patience:
                        logging.info('run out of patience\n')
                        break

        model = generate_model(self.args, pretrain=False)
        model.load_state_dict(torch.load(output_path))
        model = model.to(self.args.device)

        self.run_eval(model, train_set, val_set, 'cross-val')

    def run_eval(self, model: ResNet, ref_set: MultimodalDataset, eval_set: MultimodalDataset, test_name: Optional[str] = None) -> float:
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            ref_features = []
            ref_labels = []
            for data in tqdm(DataLoader(ref_set, batch_size=1, shuffle=False), ncols=80, desc='calculating ref'):
                ref_features.append(model.feature(data['img']))
                ref_labels.append(data['label'].item())
            ref_features = torch.cat(ref_features).to(self.args.device)
            ref_labels = torch.tensor(ref_labels).to(self.args.device)

            acc = 0
            for data in tqdm(DataLoader(eval_set, batch_size=1, shuffle=False), ncols=80, desc='evaluating'):
                feature = model.feature(data['img'].to(self.args.device))
                label = data['label'].to(self.args.device)
                loss, pred = self.forward(feature, ref_features, label, ref_labels, reg=False)
                eval_loss += loss.item()
                pred = ref_labels[pred.item()].item()
                label = label.item()
                acc += pred == label
                if test_name:
                    self.reporters[test_name].append_pred(pred, label)
        return eval_loss

if __name__ == '__main__':
    args = get_args()
    folds = load_folds(args)
    runner = Runner(args, folds)
    runner.run()
