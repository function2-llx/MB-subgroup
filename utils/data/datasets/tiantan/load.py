import itertools
import json
from argparse import Namespace
from copy import deepcopy
from typing import Optional

import numpy as np
from monai.transforms import Compose, LoadImaged, Lambdad, AddChanneld, Orientationd, Resized, ThresholdIntensityd, \
    NormalizeIntensityd, SelectItemsd
from tqdm.contrib.concurrent import process_map

from utils.dicom_utils import ScanProtocol
from utils.transforms import ConcatItemsAllowSingled

def load_info(info):
    fold_id, info = info
    label = _args.target_dict.get(info['subgroup'], None)
    if label is None:
        return None
    scans = info['scans']
    assert len(scans) == 3
    scans = {
        protocol: scans[i]
        for i, protocol in enumerate(ScanProtocol)
    }
    data = _loader(scans)
    data['label'] = label
    return fold_id, data

_loader: Optional[Compose] = None
_args: Optional[Namespace] = None

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

def load_all(args, loader=None):
    return list(itertools.chain(*load_folds(args, loader)))
