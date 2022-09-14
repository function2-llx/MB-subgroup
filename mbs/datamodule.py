from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import pandas as pd

import monai
from monai.utils import GridSampleMode
from umei.datamodule import CVDataModule, SegDataModule
from umei.utils import DataKey, DataSplit

from mbs.args import MBArgs, MBSegArgs
from mbs.utils.enums import MBDataKey, Modality, SegClass

DATASET_ROOT = Path(__file__).parent
DATA_DIR = DATASET_ROOT / 'origin'

def load_cohort():
    cohort = pd.read_excel(DATA_DIR / 'plan-split.xlsx', sheet_name='Sheet1')
    cohort.set_index('name', inplace=True)
    data = {}
    for patient, info in cohort.iterrows():
        patient = str(patient)
        patient_img_dir = DATA_DIR / 'image' / patient
        data.setdefault(str(info['split']), []).append({
            MBDataKey.CASE: patient,
            MBDataKey.SUBGROUP: info['subgroup'],
            **{
                img_type: patient_img_dir / f'{img_type}.nii'
                for img_type in list(Modality) + list(SegClass)
            },
        })

    return data

def get_classes(data: list[dict]) -> list[str]:
    return [
        x[MBDataKey.SUBGROUP]
        for x in data
    ]

class MBCVDataModule(CVDataModule):
    args: MBArgs

    def __init__(self, args: MBArgs):
        self.cohort = load_cohort()

        super().__init__(args)

    def get_cv_partitions(self):
        return [
            self.cohort[str(fold_id)]
            for fold_id in range(self.args.num_folds)
        ]

    def test_data(self) -> Sequence:
        return self.cohort[DataSplit.TEST]

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
            monai.transforms.RandRotate90D(all_keys, prob=self.args.rotate_p),
            monai.transforms.RandFlipD(all_keys, prob=self.args.flip_p, spatial_axis=0),
            monai.transforms.RandFlipD(all_keys, prob=self.args.flip_p, spatial_axis=1),
            monai.transforms.RandFlipD(all_keys, prob=self.args.flip_p, spatial_axis=2),
            monai.transforms.NormalizeIntensityD(img_keys),
            monai.transforms.LambdaD(all_keys, lambda t: t.as_tensor(), track_meta=False),
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
            monai.transforms.LambdaD(all_keys, lambda t: t.as_tensor(), track_meta=False),
            monai.transforms.ConcatItemsD(img_keys, name=DataKey.IMG),
            monai.transforms.ConcatItemsD(SegClass.ST, name=DataKey.SEG),
            monai.transforms.SelectItemsD([DataKey.IMG, DataKey.SEG, MBDataKey.CASE]),
        ])

    @property
    def test_transform(self):
        return self.val_transform
