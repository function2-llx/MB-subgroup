from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch

from utils.dicom_utils import ScanProtocol

parser = ArgumentParser()

__all__ = ['parser', 'parse_args']

# parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--train', action='store_true')
parser.add_argument('--aug', choices=['no', 'weak', 'strong'], default='weak')
parser.add_argument('--seed', type=int, default=2333)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--val_steps', type=Optional[int], default=None)
protocol_names = [value.name for value in ScanProtocol]
parser.add_argument('--protocols', nargs='+', choices=protocol_names, default=protocol_names)
parser.add_argument('--targets', choices=['all', 'G3G4', 'WS'], default='all')

def parse_args():
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.target_names = {
        'all': ['WNT', 'SHH', 'G3', 'G4'],
        'WS': ['WNT', 'SHH'],
        'G3G4': ['G3', 'G4'],
    }[args.targets]
    args.target_dict = {name: i for i, name in enumerate(args.target_names)}
    args.protocols = list(map(ScanProtocol.__getitem__, args.protocols))
    return args
