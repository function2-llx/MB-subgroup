from argparse import ArgumentParser
from pathlib import Path

import torch

from utils.dicom_utils import ScanProtocol

parser = ArgumentParser()

__all__ = ['parser', 'parse_args']

# parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--train', action='store_true')
parser.add_argument('--aug', choices=['no', 'weak', 'strong'], default='weak')
parser.add_argument('--seed', type=int, default=2333)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--debug', action='store_true')
# parser.add_argument('--n_gpu', type=int, default=2)
protocol_names = [value.name for value in ScanProtocol]
parser.add_argument('--protocols', nargs='+', choices=protocol_names, default=protocol_names)

def parse_args():
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.protocols = list(map(ScanProtocol.__getitem__, args.protocols))
    return args
