import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torchvision
from monai.transforms import *
from tqdm import tqdm

from utils.dicom_utils import Plane, ScanProtocol

_folds = None

# note: keep args.{sample_size, sample_slices} not changed among grid search process
def load_folds(args):
    global _folds
    if _folds is None:
        img_keys = list(ScanProtocol)
        loader = Compose([
            LoadImaged(img_keys),
            AddChanneld(img_keys),
            Orientationd(img_keys, axcodes='PLI'),
            SelectItemsd(img_keys),
        ])
        folds_raw = json.load(open('folds.json'))
        _folds = []

        with tqdm(total=sum(len(fold) for fold in folds_raw), ncols=80, desc='loading folds') as bar:
            for fold_raw in folds_raw:
                fold = []
                for info in fold_raw:
                    label = args.target_dict.get(info['subgroup'], None)
                    if label is not None:
                        # T1's time of echo is less than T2's, see [ref](https://radiopaedia.org/articles/t1-weighted-image)
                        scans = sorted(info['scans'], key=lambda scan: json.load(open(Path(scan.replace('.nii.gz', '.json'))))['EchoTime'])
                        assert len(scans) >= 3
                        scans = {
                            protocol: scans[i]
                            for i, protocol in enumerate(ScanProtocol)
                        }
                        data = loader(scans)
                        data['label'] = label
                        fold.append(data)
                    bar.update()
                _folds.append(fold)
    return deepcopy(_folds)

if __name__ == '__main__':
    from run import get_args
    load_folds(get_args())
