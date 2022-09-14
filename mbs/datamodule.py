from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import monai
from monai.utils import GridSampleMode
from umei.args import CVArgs
from umei.datamodule import CVDataModule, SegDataModule

from mbs.args import MBSegArgs
from mbs.utils.enums import MBDataKey, Modality, SegClass
from umei.utils import DataKey

DATASET_ROOT = Path(__file__).parent
DATA_DIR = DATASET_ROOT / 'origin'

def load_cohort():
    cohort = pd.read_excel(DATA_DIR / 'plan.xlsx', sheet_name='Sheet1')
    cohort.set_index('name', inplace=True)
    data = []
    for patient, info in cohort.iterrows():
        if info['exclude']:
            continue
        patient = str(patient)
        patient_img_dir = DATA_DIR / 'image' / patient
        data.append({
            MBDataKey.CASE: patient,
            MBDataKey.SUBGROUP: info['subgroup'],
            **{
                img_type: patient_img_dir / f'{img_type}.nii'
                for img_type in list(Modality) + list(SegClass)
            },
        })

    return data

class MBCVDataModule(CVDataModule):
    args: CVArgs

    def __init__(self, args: CVArgs):
        self.data = load_cohort()
        super().__init__(args)

    def fit_data(self) -> Sequence:
        data = self.data
        classes = [
            x[MBDataKey.SUBGROUP]
            for x in data
        ]
        return data, classes

class MBSegDataModule(MBCVDataModule, SegDataModule):
    args: MBSegArgs

    def __init__(self, args: MBSegArgs):
        super().__init__(args)

    @property
    def train_transform(self) -> Callable:
        img_keys = self.args.input_modalities
        seg_keys = [SegClass.ST]
        all_keys = img_keys + seg_keys
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(all_keys),
            monai.transforms.EnsureChannelFirstD(all_keys),
            monai.transforms.OrientationD(all_keys, axcodes='RAS'),
            monai.transforms.SpacingD(img_keys, pixdim=self.args.spacing, mode=GridSampleMode.BILINEAR),
            monai.transforms.SpacingD(seg_keys, pixdim=self.args.spacing, mode=GridSampleMode.NEAREST),
            # monai.transforms.RandRotateD(
            #     all_keys,
            #     prob=self.args.rotate_p,
            #     range_z=np.pi,
            #     mode=[GridSampleMode.BILINEAR] * len(img_keys) + [GridSampleMode.NEAREST] * len(seg_keys),
            # ),
            monai.transforms.SpatialPadD(all_keys, spatial_size=self.args.sample_shape),
            monai.transforms.RandCropByPosNegLabelD(
                all_keys,
                label_key=SegClass.ST,
                spatial_size=self.args.sample_shape,
                pos=self.args.crop_pos,
                neg=self.args.crop_neg,
                num_samples=self.args.num_crop_samples,
            ),
            monai.transforms.RandRotate90D(),
            monai.transforms.RandFlipD(all_keys, prob=self.args.flip_p, spatial_axis=0),
            monai.transforms.RandFlipD(all_keys, prob=self.args.flip_p, spatial_axis=1),
            monai.transforms.RandFlipD(all_keys, prob=self.args.flip_p, spatial_axis=2),
            monai.transforms.NormalizeIntensityD(img_keys),
            monai.transforms.ConcatItemsD(img_keys, name=DataKey.IMG),
            monai.transforms.ConcatItemsD(SegClass.ST, name=DataKey.SEG),
            monai.transforms.SelectItemsD([DataKey.IMG, DataKey.SEG]),
        ])

    @property
    def val_transform(self) -> Callable:
        img_keys = self.args.input_modalities
        all_keys = img_keys + [SegClass.ST]
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(all_keys),
            monai.transforms.EnsureChannelFirstD(all_keys),
            monai.transforms.OrientationD(all_keys, axcodes='RAS'),
            monai.transforms.SpacingD(img_keys, pixdim=self.args.spacing, mode=GridSampleMode.BILINEAR),
            monai.transforms.SpacingD(SegClass.ST, pixdim=self.args.spacing, mode=GridSampleMode.NEAREST),
            monai.transforms.NormalizeIntensityD(img_keys),
            monai.transforms.ConcatItemsD(img_keys, name=DataKey.IMG),
            monai.transforms.ConcatItemsD(SegClass.ST, name=DataKey.SEG),
            monai.transforms.SelectItemsD([DataKey.IMG, DataKey.SEG]),
        ])
