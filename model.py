import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

from utils.data import MriDataset

__all__ = ['Model']


class Model(nn.Module):
    @staticmethod
    def convert_output_to_dict(data):
        return {
            'subgroup': data[:, :4],
            'exists': data[:, 4:],
        }

    def __init__(self, ortn, train_set: MriDataset, val_set: MriDataset, args, target=None):
        super().__init__()
        self.ortn = ortn
        self.data_loaders = {
            'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn),
            'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=val_set.collate_fn),
        }
        self.args = args

        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6)

        self.optimizer = Adam(self.parameters(), lr=args.lr)
        self.weight = train_set.get_weight().to(args.device)
        self.target = target
        # weight_pos, weight_neg = train_set.get_weight()
        # self.weight_pos = nn.Parameter(weight_pos, requires_grad=False)
        # self.weight_neg = nn.Parameter(weight_neg, requires_grad=False)

        self.to(args.device)

    def train_epoch(self, epoch):
        results = {}
        for split in ['train', 'val']:
            data_loader = self.data_loaders[split]
            tot_loss = 0
            acc, tot = 0, 0
            training = split == 'train'
            self.train(training)
            with torch.set_grad_enabled(training):
                for inputs, targets in tqdm(data_loader, ncols=80, desc=f'model {self.ortn} {split} epoch {epoch}'):
                    batch_size = inputs.shape[0]
                    inputs = inputs.to(self.args.device)
                    for k, v in targets.items():
                        targets[k] = v.to(self.args.device)
                    if training:
                        self.optimizer.zero_grad()
                    logits = self.forward(inputs)
                    logits = Model.convert_output_to_dict(logits)
                    loss = F.cross_entropy(logits['exists'], targets['exists'].long()) + F.cross_entropy(logits['subgroup'], targets['subgroup'], weight=self.weight)
                    tot_loss += loss.item()

                    preds = {
                        k: logits[k].argmax(dim=1)
                        for k in logits
                    }
                    tot += batch_size
                    acc += ((targets['exists'] == preds['exists'].bool()) & (~targets['exists'] | (targets['subgroup'] == preds['subgroup']))).sum().item()

                    if training:
                        loss.backward()
                        self.optimizer.step()

            results[split] = {
                'loss': tot_loss,
                'acc': acc / tot * 100
            }
        return results

    def forward(self, inputs):
        return self.resnet.forward(inputs)
