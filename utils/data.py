import itertools
import json
import random
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Iterator, Optional

import numpy as np
import torch
from monai.data import CacheDataset
from monai.transforms import *
from torch.utils.data import Sampler
from torch.utils.data.sampler import T_co
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from utils.dicom_utils import ScanProtocol
from utils.transforms import ConcatItemsAllowSingled

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

_args: Optional[Namespace] = None
_loader: Optional[Compose] = None

def load_info(info):
    fold_id, info = info
    label = _args.target_dict.get(info['subgroup'], None)
    if label is None:
        return None
    # T1's time of echo is less than T2's, see [ref](https://radiopaedia.org/articles/t1-weighted-image)
    scans = info['scans']
    assert len(scans) == 3
    scans = {
        protocol: scans[i]
        for i, protocol in enumerate(ScanProtocol)
    }
    data = _loader(scans)
    data['label'] = label
    return fold_id, data

def load_folds(args, loader=None):
    global _args, _loader
    if loader is None:
        # keep the last some slices (default 16, discard slices of cerebrum part)
        def crop(x: np.ndarray):
            assert x.shape[2] >= args.sample_slices
            x = x[:, :, -args.sample_slices:]
            assert (x > 0).sum() > 0
            return x

        def test_clamp(x):
            assert (x > 0).sum() > 0
            assert (x < 0).sum() == 0
            return x

        loader = Compose([
            LoadImaged(args.protocols),
            Lambdad(args.protocols, crop),
            AddChanneld(args.protocols),
            Orientationd(args.protocols, axcodes='PLI'),
            Resized(args.protocols, spatial_size=(args.sample_size, args.sample_size, -1)),
            ThresholdIntensityd(args.protocols, threshold=0),
            Lambdad(args.protocols, test_clamp),
            # *[CropForegroundd(args.protocols, key) for key in args.protocols],
            NormalizeIntensityd(args.protocols, nonzero=False),
            ConcatItemsAllowSingled(args.protocols, 'img'),
            SelectItemsd('img'),
        ])
    _loader = loader
    _args = deepcopy(args)
    folds_raw = json.load(open('folds.json'))
    if args.debug:
        folds_raw = [fold[:5] for fold in folds_raw]

    folds_flattened = [(fold_id, info) for fold_id, fold_raw in enumerate(folds_raw) for info in fold_raw]
    folds_flattened = process_map(load_info, folds_flattened, ncols=80, desc='loading data', chunksize=1, max_workers=5)
    folds_flattened = filter(lambda x: x is not None, folds_flattened)
    folds = [[] for _ in range(len(folds_raw))]
    for fold_id, data in folds_flattened:
        folds[fold_id].append(data)

    return folds

def shape_stat():
    shapes_path = Path('shapes.npy')
    if shapes_path.exists():
        shapes = np.load(shapes_path)
    else:
        folds = load_folds(
            Namespace(target_dict={name: i for i, name in enumerate(['WNT', 'SHH', 'G3', 'G4'])}, debug=False),
            loader=LoadImaged(ScanProtocol),
        )
        shapes = []
        for data in tqdm(list(itertools.chain(*folds)), ncols=80):
            for key in ScanProtocol:
                img = data[key]
                shapes.append(img.shape[:2])
        shapes = np.array(shapes)
        np.save(shapes_path, shapes)
    print('shape mean: ', shapes.mean(axis=0))
    print('shape std', shapes.std(axis=0))
    print('shape median', np.median(shapes, axis=0))

if __name__ == '__main__':
    shape_stat()
