import json
from copy import deepcopy
from pathlib import Path
from typing import Optional

import monai
import monai.transforms as monai_transforms
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from finetuner_base import FinetuneArgs
from utils.transforms import CreateForegroundMaskD

dataset_root = Path(__file__).parent

loader: Optional[monai_transforms.Compose] = None
# map label string to int
target_dict: dict[str, int]
_args: FinetuneArgs

def load_case(info: pd.Series) -> dict:
    label = target_dict[info['subgroup']]
    patient = info['name(raw)']
    img_dir = _args.img_dir / patient
    seg_dir = _args.seg_dir / patient
    inputs = {}
    for protocol in _args.modalities:
        inputs[protocol] = img_dir / info[protocol.name]
    for seg in _args.seg_labels:
        inputs[seg] = seg_dir / info[seg]

    data = loader(inputs)
    data['patient'] = patient
    data['label'] = label
    return data

cohort: Optional[list[dict]] = None

# return the cohort grouped by fold, return the flattened cohort if `n_folds` is None
# for finetune
def load_cohort(args: FinetuneArgs, show_example=True, split_folds=True):
    global loader, target_dict, cohort, _args

    if cohort is None:
        _args = args
        target_dict = {
            subgroup: label
            for label, subgroup in enumerate(args.cls_labels)
        }

        from monai.utils import InterpolateMode
        loader = monai.transforms.Compose([
            monai.transforms.LoadImageD(args.modalities + args.seg_labels),
            monai.transforms.AddChannelD(args.modalities + args.seg_labels),
            monai.transforms.OrientationD(args.modalities + args.seg_labels, 'LAS'),
            monai.transforms.SpatialCropD(
                args.modalities + args.seg_labels,
                roi_slices=[slice(None), slice(None), slice(0, args.sample_slices)],
            ),
            monai.transforms.ResizeD(args.modalities, spatial_size=(args.sample_size, args.sample_size, -1)),
            monai.transforms.ResizeD(
                args.seg_labels,
                spatial_size=(args.sample_size, args.sample_size, -1),
                mode=InterpolateMode.NEAREST,
            ),
            monai.transforms.ThresholdIntensityD(args.modalities, threshold=0),
            CreateForegroundMaskD(args.modalities, mask_key='fg_mask'),
            monai.transforms.NormalizeIntensityD(args.modalities, nonzero=args.input_fg_mask),
            monai.transforms.ThresholdIntensityD(args.seg_labels, threshold=1, above=False, cval=1),
        ])
        cohort_info = read_cohort_info(args.cls_labels)
        cohort = []
        # for _, info in cohort_info.iterrows():
        #     cohort.append(load_case(info))
        cohort = process_map(load_case, [info for _, info in cohort_info.iterrows()], desc='loading cohort', ncols=80, max_workers=16)
        if show_example:
            import random
            from matplotlib import pyplot as plt
            example = random.choice(cohort)
            fig, ax = plt.subplots(1, 3)
            ax = {
                protocol: ax[i]
                for i, protocol in enumerate(args.modalities)
            }
            from utils.dicom_utils import ScanProtocol
            for protocol in args.modalities:
                img = example[protocol][0]
                idx = img.shape[2] // 2
                ax[protocol].imshow(np.rot90(img[:, :, idx]), cmap='gray')
                seg_t = {
                    ScanProtocol.T2: 'AT',
                    ScanProtocol.T1c: 'CT',
                }.get(protocol, None)
                if seg_t is not None and seg_t in args.seg_labels:
                    from matplotlib.colors import listedColormap
                    ax[protocol].imshow(np.rot90(example[seg_t][0, :, :, idx]), vmin=0, vmax=1, cmap=listedColormap(['none', 'green']), alpha=0.5)

            plt.show()
            plt.close()

    if split_folds:
        folds = json.load(open(dataset_root / args.folds_file))
        folds = list(map(set, folds))
        return [[sample for sample in cohort if sample['patient'] in fold] for fold in folds]
    else:
        return deepcopy(cohort)

def read_cohort_info(subgroups: Optional[list[str]] = None) -> pd.DataFrame:
    global cohort
    cohort = pd.read_excel(dataset_root / 'cohort.xlsx')
    if subgroups is not None:
        cohort = cohort[cohort['subgroup'].isin(subgroups)]
    return cohort
