from argparse import ArgumentParser
from typing import Optional

import torch

parser = ArgumentParser(add_help=False)

parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--train', action='store_true')
parser.add_argument('--aug', choices=['no', 'weak', 'strong'], default='weak')
parser.add_argument('--crop_ratio', type=float, default=1)
parser.add_argument('--lr_reduce_factor', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=2333)
parser.add_argument('--patience', type=int, default=6)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--val_steps', type=Optional[int], default=None)
parser.add_argument('--force_retrain', action='store_true')

def process_args(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.rank = 0
    return args
