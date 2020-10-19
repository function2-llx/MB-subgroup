import os
import csv
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import load_data
from preprocess import splits

parser = ArgumentParser()
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--val_batch_size', type=int, default=8)

args = parser.parse_args()

device = torch.device(args.device)
batch_size = {
    'train': args.train_batch_size,
    'val': args.val_batch_size,
}


if __name__ == '__main__':
    datasets = load_data()
    data_loaders = {split: DataLoader(datasets[split], batch_size[split], split == 'train') for split in splits}
    model = torch.hub.load('pytorch/vision:v0.7.0', 'resnet18', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 4)
