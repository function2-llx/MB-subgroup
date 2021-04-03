from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch

from utils.dicom_utils import ScanProtocol

parser = ArgumentParser(add_help=False)

# parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--train', action='store_true')
parser.add_argument('--aug', choices=['no', 'weak', 'strong'], default='weak')
parser.add_argument('--lr_reduce_factor', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=2333)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--val_steps', type=Optional[int], default=None)

