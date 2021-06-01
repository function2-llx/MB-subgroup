import itertools
import json
from argparse import Namespace
from copy import deepcopy
from typing import Optional, Dict

import numpy as np
import monai.transforms as monai_transforms
from tqdm.contrib.concurrent import process_map

from utils.data.datasets.tiantan import dataset_dir
from utils.data.datasets.tiantan.preprocess import output_dir
from utils.dicom_utils import ScanProtocol

_loader: Optional[monai_transforms.Compose] = None
# map label string to int
_target_dict: Dict[str, int]

def load_info(info):
    # fold_id, info = info
    label = _target_dict[info['subgroup']]
    scans = info['scans']
    assert len(scans) == 3
    scans = {
        protocol: str(dataset_dir / output_dir / info['patient'] / f'{scans[i]}_ss.nii.gz')
        for i, protocol in enumerate(ScanProtocol)
    }
    data = _loader(scans)
    data['label'] = label
    data['patient'] = info['patient']
    return data

# def load_folds(args, loader=None):
#     global _args, _loader
#     if loader is None:
#         loader = monai_transforms.Compose([
#             monai_transforms.LoadImaged(args.protocols),
#             monai_transforms.Lambdad(args.protocols, crop),
#             monai_transforms.AddChanneld(args.protocols),
#             monai_transforms.Orientationd(args.protocols, axcodes='LAS'),
#             monai_transforms.ThresholdIntensityd(args.protocols, threshold=0),
#
#             monai_transforms.SelectItemsd(args.protocols),
#             # Resized(args.protocols, spatial_size=(args.sample_size, args.sample_size, -1)),
#             # NormalizeIntensityd(args.protocols, nonzero=True),
#             # ConcatItemsd(args.protocols, 'img'),
#         ])
#     _loader = loader
#     _args = deepcopy(args)
#     folds_raw = [
#         list(filter(lambda info: info['subgroup'] in args.target_dict, fold_raw))
#         for fold_raw in json.load(open(dataset_dir / f'folds-{args.n_folds}.json'))
#     ]
#     if args.debug:
#         folds_raw = [fold[:5] for fold in folds_raw]
#
#     folds_flattened = [(fold_id, info) for fold_id, fold_raw in enumerate(folds_raw) for info in fold_raw]
#     folds_flattened = process_map(load_info, folds_flattened, ncols=80, desc='loading data')
#     folds = [[] for _ in range(len(folds_raw))]
#     for fold_id, data in folds_flattened:
#         folds[fold_id].append(data)
#
#     return folds


def load_cohort(args, n_folds=None):
    global _args, _loader
    cohort = json.load(open(dataset_dir / 'cohort.json'))
    _args = deepcopy(args)
    _loader = monai_transforms.Compose([
        monai_transforms.LoadImaged(args.protocols),
        monai_transforms.AddChanneld(args.protocols),
        monai_transforms.ThresholdIntensityd(args.protocols, threshold=0),

        monai_transforms.SelectItemsd(args.protocols),
        # Resized(args.protocols, spatial_size=(args.sample_size, args.sample_size, -1)),
        # NormalizeIntensityd(args.protocols, nonzero=True),
        # ConcatItemsd(args.protocols, 'img'),
    ])
    data = process_map(load_info, cohort)
    if n_folds is not None:
        folds = json.load(open(dataset_dir / f'folds-{n_folds}.json'))
        pass
    return data
