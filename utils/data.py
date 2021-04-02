import itertools
import json
import random
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from monai.data import CacheDataset
from monai.transforms import *
from torch.utils.data import Sampler
from torch.utils.data.sampler import T_co
from tqdm.contrib.concurrent import process_map

from utils.dicom_utils import ScanProtocol
from utils.to_tensor import ToTensorDeviced

class MultimodalDataset(CacheDataset):
    def __init__(self, data: List[Dict], transform: Transform, num_classes: int):
        super().__init__(data, transform, progress=False)
        self.num_classes = num_classes
        self.labels = [data['label'] for data in self.data]

    def get_weight(self, strategy) -> torch.Tensor:
        if strategy == 'equal':
            return torch.ones(self.num_classes)
        elif strategy == 'invsqrt':
            weight = torch.zeros(self.num_classes)
            for label in self.labels:
                weight[label] += 1
            weight.sqrt_()
            weight = weight.sum() / weight
            weight = weight / weight.sum()
            return weight
        elif strategy == 'inv':
            weight = torch.zeros(self.num_classes)
            for label in self.labels:
                weight[label] += 1
            # weight.sqrt_()
            weight = weight.sum() / weight
            weight = weight / weight.sum()
            return weight
        else:
            raise ValueError(f'unsupported weight strategy of {strategy}')

class BalancedSampler(Sampler):
    def __init__(self, dataset: MultimodalDataset, total: Optional[int] = None):
        super().__init__(dataset)
        self.indexes = {}
        for i, data in enumerate(dataset):
            self.indexes.setdefault(data['label'], []).append(i)
        self.labels = list(self.indexes.keys())

        if total is None:
            self.total = len(self.labels) * max([len(v) for v in self.indexes.values()])
        else:
            self.total = total

    def __iter__(self) -> Iterator[T_co]:
        pointers = {label: len(indexes) - 1 for label, indexes in self.indexes.items()}
        label_id = 0
        for _ in range(self.total):
            label = self.labels[label_id]
            pointers[label] += 1
            if pointers[label] == len(self.indexes[label]):
                pointers[label] = 0
                random.shuffle(self.indexes[label])
            yield self.indexes[label][pointers[label]]
            label_id = (label_id + 1) % len(self.labels)

    def __len__(self):
        return self.total

_folds = None
_args: Optional[Namespace] = None
modalities = list(ScanProtocol)
loader = Compose([
    LoadImaged(modalities),
    NormalizeIntensityd(modalities, nonzero=False),
    AddChanneld(modalities),
    Orientationd(modalities, axcodes='PLI'),
])

def load_info(info):
    fold_id, info = info
    label = _args.target_dict.get(info['subgroup'], None)
    if label is None:
        return None
    # T1's time of echo is less than T2's, see [ref](https://radiopaedia.org/articles/t1-weighted-image)
    scans = sorted(info['scans'], key=lambda scan: json.load(open(Path(scan.replace('.nii.gz', '.json'))))['EchoTime'])
    assert len(scans) >= 3
    scans = {
        protocol: scans[i]
        for i, protocol in enumerate(ScanProtocol)
    }
    data = loader(scans)
    data['label'] = label
    return fold_id, data

def load_folds(args):
    global _folds, _args
    if _folds is None:
        _args = deepcopy(args)
        folds_raw = json.load(open('folds.json'))
        if args.debug:
            folds_raw = [fold[:5] for fold in folds_raw]

        folds_flattened = [(fold_id, info) for fold_id, fold_raw in enumerate(folds_raw) for info in fold_raw]
        folds_flattened = process_map(load_info, folds_flattened, ncols=80, desc='loading data', chunksize=1, max_workers=5)
        folds_flattened = filter(lambda x: x is not None, folds_flattened)
        _folds = [[] for _ in range(len(folds_raw))]
        for fold_id, data in folds_flattened:
            _folds[fold_id].append(data)

    return deepcopy(_folds)

if __name__ == '__main__':
    from run_3dresnet import get_args
    args = get_args()
    load_folds(args)
