import itertools
import json
from argparse import Namespace
from copy import deepcopy
from typing import Optional

import numpy as np
import monai.transforms as monai_transforms
from tqdm.contrib.concurrent import process_map

from utils.data.datasets.tiantan import dataset_dir
from utils.data.datasets.tiantan.preprocess import output_dir
from utils.dicom_utils import ScanProtocol

def load_info(info):
    fold_id, info = info
    label = _args.target_dict[info['subgroup']]
    scans = info['scans']
    assert len(scans) == 3
    scans = {
        protocol: str(dataset_dir / output_dir / info['patient'] / f'{scans[i]}_ss.nii.gz')
        for i, protocol in enumerate(ScanProtocol)
    }
    data = _loader(scans)
    data['label'] = label
    data['patient'] = info['patient']
    return fold_id, data

_loader: Optional[monai_transforms.Compose] = None
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

        # loader = Compose([
        #     LoadImaged(args.protocols),
        #     Lambdad(args.protocols, crop),
        #     AddChanneld(args.protocols),
        #     Orientationd(args.protocols, axcodes='PLI'),
        #     ThresholdIntensityd(args.protocols, threshold=0),
        #     Resized(args.protocols, spatial_size=(args.sample_size, args.sample_size, -1)),
        #     NormalizeIntensityd(args.protocols, nonzero=True),
        #     ConcatItemsd(args.protocols, 'img'),
        #     SelectItemsd('img'),
        # ])
        loader = monai_transforms.Compose([
            monai_transforms.LoadImaged(args.protocols),
            monai_transforms.Lambdad(args.protocols, crop),
            monai_transforms.AddChanneld(args.protocols),
            monai_transforms.Orientationd(args.protocols, axcodes='LAS'),
            monai_transforms.ThresholdIntensityd(args.protocols, threshold=0),

            monai_transforms.SelectItemsd(args.protocols),
            # Resized(args.protocols, spatial_size=(args.sample_size, args.sample_size, -1)),
            # NormalizeIntensityd(args.protocols, nonzero=True),
            # ConcatItemsd(args.protocols, 'img'),
        ])
    _loader = loader
    _args = deepcopy(args)
    folds_raw = [
        list(filter(lambda info: info['subgroup'] in args.target_dict, fold_raw))
        for fold_raw in json.load(open(dataset_dir / f'folds-{args.n_folds}.json'))
    ]
    if args.debug:
        folds_raw = [fold[:5] for fold in folds_raw]

    folds_flattened = [(fold_id, info) for fold_id, fold_raw in enumerate(folds_raw) for info in fold_raw]
    folds_flattened = process_map(load_info, folds_flattened, ncols=80, desc='loading data')
    folds = [[] for _ in range(len(folds_raw))]
    for fold_id, data in folds_flattened:
        folds[fold_id].append(data)

    return folds

def load_all(args, loader=None):
    return list(itertools.chain(*load_folds(args, loader)))
