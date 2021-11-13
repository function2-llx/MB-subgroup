import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch

import monai.transforms as monai_transforms
import pandas as pd
from tqdm.contrib.concurrent import process_map

from utils.args import FinetuneArgs
from utils.conf import Conf
# from .check_files import data_dir

dataset_root = Path(__file__).parent

loader: Optional[monai_transforms.Compose] = None
# map label string to int
target_dict: Dict[str, int]
_args: Conf

def load_info(info: pd.Series):
    label = target_dict[info['subgroup']]
    patient = info['name(raw)']
    img_dir = _args.img_dir / patient
    seg_dir = _args.seg_dir / patient
    inputs = {}
    for protocol in _args.protocols:
        inputs[protocol] = img_dir / info[protocol.name]
    for seg in _args.segs:
        inputs[seg] = seg_dir / info[seg]

    data = loader(inputs)
    data['patient'] = patient
    data['label'] = label
    return data

cohort: Optional[List[Dict]] = None

# return the cohort grouped by fold, return the flattened cohort if `n_folds` is None
def load_cohort(args: FinetuneArgs, show_example=True):
    global loader, target_dict, cohort, _args

    if cohort is None:
        _args = args
        target_dict = {
            subgroup: label
            for label, subgroup in enumerate(args.subgroups)
        }

        from monai.utils import InterpolateMode
        loader = monai_transforms.Compose([
            monai_transforms.LoadImageD(args.protocols + args.segs),
            monai_transforms.AddChannelD(args.protocols + args.segs),
            monai_transforms.OrientationD(args.protocols + args.segs, 'LAS'),
            monai_transforms.SpatialCropD(args.protocols + args.segs, roi_slices=[slice(None), slice(None), slice(0, args.sample_slices)]),
            monai_transforms.ResizeD(args.protocols, spatial_size=(args.sample_size, args.sample_size, -1)),
            monai_transforms.ResizeD(
                args.segs,
                spatial_size=(args.sample_size, args.sample_size, -1),
                mode=InterpolateMode.NEAREST,
            ),
            monai_transforms.ConcatItemsD(args.protocols, 'img'),
            monai_transforms.ConcatItemsD(args.segs, 'seg'),
            monai_transforms.CastToTypeD('seg', dtype=torch.int),
            # monai_transforms.SelectItemsD(['img', 'seg']),
            monai_transforms.ThresholdIntensityD('img', threshold=0),
            monai_transforms.NormalizeIntensityD('img', channel_wise=True, nonzero=True),
        ])
        cohort = pd.read_excel(dataset_root / 'cohort.xlsx')
        cohort = cohort[cohort['subgroup'].isin(args.subgroups)]
        cohort = process_map(load_info, [info for _, info in cohort.iterrows()], desc='loading cohort', ncols=80, max_workers=16)
        if show_example:
            import random
            from matplotlib import pyplot as plt
            example = random.choice(cohort)
            fig, ax = plt.subplots(1, 3)
            ax = {
                protocol: ax[i]
                for i, protocol in enumerate(args.protocols)
            }
            from utils.dicom_utils import ScanProtocol
            for protocol in args.protocols:
                img = example[protocol][0]
                idx = img.shape[2] // 2
                ax[protocol].imshow(np.rot90(img[:, :, idx]), cmap='gray')
                seg_t = {
                    ScanProtocol.T2: 'AT',
                    ScanProtocol.T1c: 'CT',
                }.get(protocol, None)
                if seg_t is not None:
                    from matplotlib.colors import ListedColormap
                    ax[protocol].imshow(np.rot90(example[seg_t][0, :, :, idx]), vmin=0, vmax=1, cmap=ListedColormap(['none', 'green']), alpha=0.5)

            plt.show()

    if args.folds_file is not None:
        folds = json.load(open(dataset_root / args.folds_file))
        folds = list(map(set, folds))
        return [[sample for sample in cohort if sample['patient'] in fold] for fold in folds]
    else:
        return deepcopy(cohort)

def main():
    pass
    # from utils.conf import get_conf
    # load_cohort(get_conf())

if __name__ == '__main__':
    main()
