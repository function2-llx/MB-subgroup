import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet50 as resnet
from tqdm import tqdm

from utils.data import MriDataset

__all__ = ['Model']


class Model(nn.Module):
    @staticmethod
    def get_exists(output):
        return output[4:6].argmax().item()

    @staticmethod
    def get_subtype(output):
        return output[:4].argmax().item()

    def __init__(self, ortn, train_set: MriDataset, val_set: MriDataset, args):
        super().__init__()
        self.ortn = ortn
        self.data_loaders = {
            'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn),
            'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=val_set.collate_fn),
        }
        self.args = args

        self.resnet = resnet(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6)

        self.optimizer = Adam(self.parameters(), lr=args.lr)
        self.weight = train_set.get_weight().to(args.device)
        # weight_pos, weight_neg = train_set.get_weight()
        # self.weight_pos = nn.Parameter(weight_pos, requires_grad=False)
        # self.weight_neg = nn.Parameter(weight_neg, requires_grad=False)

        self.to(self.args.device)

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
                    inputs = inputs.to(self.args.device)
                    for k, v in targets.items():
                        targets[k] = v.to(self.args.device)
                    if training:
                        self.optimizer.zero_grad()
                    logits = self.resnet.forward(inputs)
                    loss = F.cross_entropy(logits[:, 4:], targets['exists'].long()) + F.cross_entropy(logits[:, :4], targets['subtype'], weight=self.weight)
                    tot_loss += loss.item()

                    # outputs = torch.sigmoid(logits)
                    for exists, subtype, logit in zip(targets['exists'].tolist(), targets['subtype'].tolist(), logits):
                        tot += 1
                        if exists == self.get_exists(logit) and (not exists or subtype == self.get_subtype(logit)):
                            acc += 1

                    if training:
                        loss.backward()
                        self.optimizer.step()
            results[split] = {
                'loss': tot_loss,
                'acc': acc / tot * 100
            }
        return results
