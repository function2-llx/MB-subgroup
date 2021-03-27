import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Optional

import torch
from monai.data import CacheDataset
from monai.transforms import *
from tqdm.contrib.concurrent import process_map

from utils.dicom_utils import ScanProtocol


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
        _folds = [[] for _ in range(len(args.target_names))]
        for fold_id, data in folds_flattened:
            _folds[fold_id].append(data)

    return deepcopy(_folds)

if __name__ == '__main__':
    from run_3d import get_args
    args = get_args()
    load_folds(args)
