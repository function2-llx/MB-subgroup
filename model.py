import os
import json
import shutil
from typing import *

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet101
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.data import ImageRecognitionDataset, targets
from utils.report import ClassificationReporter

__all__ = ['ImageRecognitionModel']


class ImageRecognitionModel(nn.Module):
    def __init__(self, ortn, args, target, load_weights=False):
        super().__init__()
        self.ortn = ortn
        self.args = args
        self.target = target
        self.target_names = targets[target]

        self.prefix = os.path.join('lr={lr},bs={batch_size}'.format(**args.__dict__), ortn, target)
        self.output_dir = os.path.join('output', self.prefix)
        os.makedirs(self.output_dir, exist_ok=True)

        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, len(self.target_names))
        if load_weights:
            load_path = os.path.join(self.output_dir, 'checkpoint.pth.tar')
            self.load_state_dict(torch.load(load_path))
            print(f'load weights of {ortn}-{target} from {load_path}')

        self.to(args.device)

    def run_train(self, datasets: Dict[str, ImageRecognitionDataset]):
        optimizer = Adam(self.parameters(), lr=self.args.lr)
        weight = datasets['train'].get_weight().to(self.args.device)
        data_loaders = {
            split: DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=(split == 'train'),
                collate_fn=dataset.collate_fn
            )
            for split, dataset in datasets.items()
        }

        writer = SummaryWriter(os.path.join('runs', self.prefix))
        best_acc = 0
        patience = 0
        for epoch in range(1, self.args.epochs + 1):
            results = self.train_epoch(epoch, weight, data_loaders, optimizer)
            print(json.dumps(results, indent=4, ensure_ascii=False))
            for split, result in results.items():
                for k, v in result.items():
                    writer.add_scalar(f'{split}/{k}', v, epoch)
            val_acc = results['val']['acc']
            if best_acc < val_acc:
                best_acc = val_acc
                torch.save(self.state_dict(), os.path.join(self.output_dir, 'checkpoint-train.pth.tar'))
                patience = 0
            else:
                patience += 1
                print(f'patience {patience}/{self.args.patience}')
                if patience >= self.args.patience:
                    print('run out of patience')
                    break
            writer.flush()
        shutil.copy(
            os.path.join(self.output_dir, 'checkpoint-train.pth.tar'),
            os.path.join(self.output_dir, 'checkpoint.pth.tar'),
        )

    def train_epoch(self, epoch, weight, data_loaders, optimizer):
        results = {}
        for split in ['train', 'val']:
            data_loader = data_loaders[split]
            tot_loss = 0
            acc, tot = 0, 0
            training = split == 'train'
            self.train(training)
            with torch.set_grad_enabled(training):
                for _, inputs, labels in tqdm(data_loader, ncols=80, desc=f'model {self.ortn} {self.target} {split} epoch {epoch}'):
                    inputs = inputs.to(self.args.device)
                    labels = labels.to(self.args.device)
                    if training:
                        optimizer.zero_grad()
                    logits = self.forward(inputs)
                    loss = F.cross_entropy(logits, labels, weight=weight)
                    tot_loss += loss.item()
                    if training:
                        loss.backward()
                        optimizer.step()
                    preds = logits.argmax(dim=1)
                    tot += inputs.shape[0]
                    acc += (labels == preds).sum().item()

            results[split] = {
                'loss': tot_loss,
                'acc': acc / tot * 100
            }
        return results

    def run_test(self, test_set: ImageRecognitionDataset):
        reporter = ClassificationReporter(self.output_dir, self.target_names)
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, collate_fn=test_set.collate_fn)
        self.eval()
        with torch.no_grad():
            for patients, inputs, labels in tqdm(test_loader, ncols=80, desc=f'testing {self.ortn} {self.target}'):
                inputs = inputs.to(self.args.device)
                logits = self.forward(inputs)
                for patient, logit, label in zip(patients, logits, labels):
                    reporter.append(patient, logit, label.item())

        reporter.report()
        reporter.plot_roc()

    def forward(self, inputs):
        return self.resnet.forward(inputs)
