import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch

import monai.transforms as monai_transforms
import pandas as pd
from tqdm.contrib.concurrent import process_map

from utils.conf import Conf
# from .check_files import data_dir

dataset_root = Path(__file__).parent

loader: Optional[monai_transforms.Compose] = None
# map label string to int
target_dict: Dict[str, int]
_conf: Conf

def load_info(info: pd.Series):
    label = target_dict[info['subgroup']]
    patient = info['name(raw)']
    img_dir = _conf.img_dir / patient
    seg_dir = _conf.seg_dir / patient
    inputs = {}
    for protocol in _conf.protocols:
        inputs[protocol] = img_dir / info[protocol.name]
    for seg in _conf.segs:
        inputs[seg] = seg_dir / info[seg]

    data = loader(inputs)
    data['patient'] = patient
    data['label'] = label
    return data

cohort: Optional[List[Dict]] = None

# return the cohort grouped by fold, return the flattened cohort if `n_folds` is None
def load_cohort(conf: Conf, show_example=True):
    global loader, target_dict, cohort, _conf

    if cohort is None:
        _conf = conf
        target_dict = {
            subgroup: label
            for label, subgroup in enumerate(conf.subgroups)
        }

        from monai.utils import InterpolateMode
        loader = monai_transforms.Compose([
            monai_transforms.LoadImageD(conf.protocols + conf.segs),
            monai_transforms.AddChannelD(conf.protocols + conf.segs),
            monai_transforms.OrientationD(conf.protocols + conf.segs, 'LAS'),
            monai_transforms.SpatialCropD(conf.protocols + conf.segs, roi_slices=[slice(None), slice(None), slice(0, conf.sample_slices)]),
            monai_transforms.ResizeD(conf.protocols, spatial_size=(conf.sample_size, conf.sample_size, -1)),
            monai_transforms.ResizeD(
                conf.segs,
                spatial_size=(conf.sample_size, conf.sample_size, -1),
                mode=InterpolateMode.NEAREST,
            ),
            monai_transforms.ConcatItemsD(conf.protocols, 'img'),
            monai_transforms.ConcatItemsD(conf.segs, 'seg'),
            monai_transforms.CastToTypeD('seg', dtype=torch.int),
            # monai_transforms.SelectItemsD(['img', 'seg']),
            monai_transforms.ThresholdIntensityD('img', threshold=0),
            monai_transforms.NormalizeIntensityD('img', channel_wise=True, nonzero=True),
        ])
        cohort = pd.read_excel(dataset_root / 'cohort.xlsx')
        cohort = cohort[cohort['subgroup'].isin(conf.subgroups)]
        cohort = process_map(load_info, [info for _, info in cohort.iterrows()], desc='loading cohort', ncols=80, max_workers=16)
        if show_example:
            import random
            from matplotlib import pyplot as plt
            example = random.choice(cohort)
            fig, ax = plt.subplots(1, 3)
            ax = {
                protocol: ax[i]
                for i, protocol in enumerate(conf.protocols)
            }
            from utils.dicom_utils import ScanProtocol
            for protocol in conf.protocols:
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

    if conf.n_folds > 1:
        folds = json.load(open(dataset_root / conf.folds_file))
        folds = list(map(set, folds))
        return [[sample for sample in cohort if sample['patient'] in fold] for fold in folds]
    else:
        return deepcopy(cohort)

def main():
    from utils.conf import get_conf
    load_cohort(get_conf())

if __name__ == '__main__':
    main()
