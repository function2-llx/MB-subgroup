import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torchvision.models import resnet18 as resnet
from ignite.contrib.metrics.roc_auc import roc_auc_compute_fn


class Model(nn.Module):
    def __init__(self, ortn, train_set, val_set, args):
        super().__init__()
        self.ortn = ortn
        self.data_loaders = {
            'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
            'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False),
        }
        self.args = args

        self.resnet = resnet(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 4),
            nn.Sigmoid(),
        ).to(self.args.device)

        self.optimizer = Adam(self.parameters(), lr=args.lr)
        # self.loss_fn = nn.BCELoss()

        self.to(self.args.device)

    def train_epoch(self, epoch):
        results = {}
        for split, data_loader in self.data_loaders.items():
            tot_loss = 0
            acc, tot = 0, 0
            training = split == 'train'
            self.train(training)
            with torch.set_grad_enabled(training):
                for inputs, labels in tqdm(data_loader, ncols=80, desc=f'model {self.ortn} {split} epoch {epoch}'):
                    inputs = inputs.to(self.args.device)
                    labels = labels.to(self.args.device)
                    singel_labels = labels.argmax(dim=1)
                    if training:
                        self.optimizer.zero_grad()
                    outputs = self.resnet.forward(inputs)
                    weights = torch.full(outputs.shape, 1).to(self.args.device)
                    # for i, label in enumerate(labels):
                    #     label = label.argmax()
                    #     weights[i, label] = 1
                    loss = F.binary_cross_entropy(outputs, labels, weights)
                    preds = (outputs > 0.5).to(torch.long)
                    single_preds = outputs.argmax(dim=1)
                    tot_loss += loss.item()
                    for label, pred in zip(singel_labels, single_preds):
                        tot += 1
                        acc += torch.equal(label, pred)
                    if training:
                        loss.backward()
                        self.optimizer.step()
            results[split] = {
                'loss': tot_loss,
                'acc': acc / tot * 100
            }
        return results
