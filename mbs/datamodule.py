from collections.abc import Callable
from functools import cached_property
import itertools
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

import monai
from monai.data import CacheDataset, MetaTensor
from monai.utils import GridSampleMode
from umei.datamodule import CVDataModule
from umei.utils import DataKey, DataSplit

from mbs.args import MBArgs, MBSegArgs
from mbs.utils.enums import MBDataKey, Modality, SUBGROUPS, SegClass
from mbs.transforms import CropBBoxCenterD

DATASET_ROOT = Path(__file__).parent
DATA_DIR = DATASET_ROOT / 'origin'

def load_cohort():
    cohort = pd.read_excel(DATA_DIR / 'plan-split.xlsx', sheet_name='Sheet1').set_index('name')
    clinical = pd.read_excel(DATA_DIR / 'clinical-com.xlsx', dtype=str).set_index('住院号')
    data = {}
    for patient, info in cohort.iterrows():
        patient = str(patient)
        num = patient[:6]
        sex: str = clinical.loc[num, 'sex']
        sex_vec = torch.zeros(2)
        if sex.lower() == 'f':
            sex_vec[0] = 1
        if sex.lower() == 'm':
            sex_vec[1] = 1
        age = clinical.loc[num, 'age']
        age_vec = torch.zeros(2)
        if not pd.isna(age):
            age_vec[0] = 1
            age_vec[1] = parse_age(age) / 100
        clinical_vec = torch.cat([sex_vec, age_vec])

        patient_img_dir = DATA_DIR / info['group'] / patient
        data.setdefault(str(info['split']), []).append({
            MBDataKey.CASE: patient,
            MBDataKey.SUBGROUP_ID: SUBGROUPS.index(info['subgroup']),
            **{
                img_type: patient_img_dir / f'{img_type}.nii'
                for img_type in list(Modality) + list(SegClass)
            },
            DataKey.CLINICAL: clinical_vec,
        })

    return data

def get_classes(data: list[dict]) -> list:
    return [
        x[MBDataKey.SUBGROUP_ID]
        for x in data
    ]

class MBCVDataModule(CVDataModule):
    args: MBSegArgs

    def __init__(self, args: MBSegArgs):
        super().__init__(args)

    @cached_property
    def cohort(self):
        return load_cohort()

    @cached_property
    def partitions(self):
        ret = [
            self.cohort[str(fold_id)]
            for fold_id in range(self.args.num_folds)
        ]
        # trick: select training data for fold-i is by deleting the i-th item
        if self.args.include_adults:
            ret.append(self.cohort[DataSplit.TRAIN])
        return ret

    def test_data(self) -> Sequence:
        return self.cohort[DataSplit.TEST]

    def all_data(self) -> Sequence:
        return list(itertools.chain(*self.cohort.values()))

class MBSegDataModule(MBCVDataModule):
    args: MBSegArgs

    @property
    def train_transform(self) -> Callable:
        img_keys = self.args.input_modalities
        seg_keys = self.args.seg_classes
        all_keys = img_keys + seg_keys
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(all_keys),
            monai.transforms.EnsureChannelFirstD(all_keys),
            monai.transforms.OrientationD(all_keys, axcodes='RAS'),
            monai.transforms.SpacingD(img_keys, pixdim=self.args.spacing, mode=GridSampleMode.BILINEAR),
            monai.transforms.SpacingD(seg_keys, pixdim=self.args.spacing, mode=GridSampleMode.NEAREST),
            monai.transforms.ResizeWithPadOrCropD(all_keys, spatial_size=self.args.pad_crop_size),
            monai.transforms.RandCropByPosNegLabelD(
                all_keys,
                label_key=SegClass.ST,
                spatial_size=self.args.sample_shape,
                pos=self.args.crop_pos,
                neg=self.args.crop_neg,
                num_samples=self.args.num_crop_samples,
            ),
            monai.transforms.RandFlipD(all_keys, prob=self.args.flip_p, spatial_axis=0),
            monai.transforms.RandFlipD(all_keys, prob=self.args.flip_p, spatial_axis=1),
            monai.transforms.RandFlipD(all_keys, prob=self.args.flip_p, spatial_axis=2),
            monai.transforms.RandRotate90D(all_keys, prob=self.args.rotate_p),
            monai.transforms.NormalizeIntensityD(img_keys),
            monai.transforms.LambdaD(all_keys, lambda t: t.as_tensor(), track_meta=False),
            monai.transforms.ConcatItemsD(img_keys, name=DataKey.IMG),
            monai.transforms.ConcatItemsD(seg_keys, name=DataKey.SEG),
            monai.transforms.SelectItemsD([DataKey.IMG, DataKey.SEG]),
        ])

    @property
    def val_transform(self) -> Callable:
        img_keys = self.args.input_modalities
        seg_keys = self.args.seg_classes
        all_keys = img_keys + seg_keys
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(all_keys),
            monai.transforms.EnsureChannelFirstD(all_keys),
            monai.transforms.OrientationD(all_keys, axcodes='RAS'),
            monai.transforms.SpacingD(img_keys, pixdim=self.args.spacing, mode=GridSampleMode.BILINEAR),
            monai.transforms.SpacingD(seg_keys, pixdim=self.args.spacing, mode=GridSampleMode.NEAREST),
            monai.transforms.ResizeWithPadOrCropD(all_keys, spatial_size=self.args.pad_crop_size),
            monai.transforms.NormalizeIntensityD(img_keys),
            monai.transforms.LambdaD(all_keys, lambda t: t.as_tensor(), track_meta=False),
            monai.transforms.ConcatItemsD(img_keys, name=DataKey.IMG),
            monai.transforms.ConcatItemsD(seg_keys, name=DataKey.SEG),
            monai.transforms.SelectItemsD([DataKey.IMG, DataKey.SEG, MBDataKey.CASE]),
        ])

    @property
    def test_transform(self):
        return self.val_transform

class MBDataModule(MBCVDataModule):
    args: MBArgs

    def __init__(self, args: MBArgs):
        super().__init__(args)
        suffix = f'th{args.th}'
        if args.do_post:
            suffix += '-post'
        for x in itertools.chain(*self.cohort.values()):
            seg_pred_dir = args.seg_pred_dir / x[MBDataKey.CASE] / suffix
            for seg_cls in args.seg_classes:
                x[f'{seg_cls}-pred'] = seg_pred_dir / f'{seg_cls}.nii.gz'
                # x[f'{seg_cls}-pred'] = DATA_DIR / 'image' / x[MBDataKey.CASE] / f'{seg_cls}.nii'

        if args.cls_weights is None:
            cls_cnt = np.zeros(args.num_cls_classes, dtype=np.float64)
            for i in range(args.num_folds):
                for x in self.cohort[str(i)]:
                    cls_cnt[x[DataKey.CLS]] += 1
            args.cls_weights = (1. / cls_cnt).tolist()

    @cached_property
    def cohort(self):
        cohort = super().cohort
        for fold in cohort.values():
            for x in fold:
                subgroup = SUBGROUPS[x[MBDataKey.SUBGROUP_ID]]
                x[DataKey.CLS] = self.args.cls_map[subgroup]
                if not self.args.use_clinical:
                    x.pop(DataKey.CLINICAL)
        return {
            k: list(filter(lambda x: x[DataKey.CLS] != -1, fold))
            for k, fold in cohort.items()
        }

    @property
    def train_transform(self):
        img_keys = self.args.input_modalities
        seg_keys = self.args.seg_classes
        pred_keys = [f'{seg_cls}-pred' for seg_cls in self.args.seg_classes]
        bbox_src_key = f'{self.args.crop_ref}-pred'
        all_keys = img_keys + seg_keys + pred_keys
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(all_keys),
            monai.transforms.EnsureChannelFirstD(all_keys),
            monai.transforms.OrientationD(all_keys, axcodes='RAS'),
            monai.transforms.SpacingD(img_keys, pixdim=self.args.spacing, mode=GridSampleMode.BILINEAR),
            monai.transforms.SpacingD(seg_keys, pixdim=self.args.spacing, mode=GridSampleMode.NEAREST),
            monai.transforms.ResizeWithPadOrCropD(all_keys, spatial_size=self.args.pad_crop_size),
            CropBBoxCenterD(all_keys, bbox_src_key, crop_size=self.args.sample_shape),
            # monai.transforms.CropForegroundD(all_keys, source_key=f'{SegClass.ST}-pred'),
            # monai.transforms.MaskIntensityD(all_keys, mask_key=f'{SegClass.ST}-pred'),
            # monai.transforms.RandSpatialCropD(
            #     all_keys,
            #     roi_size=self.args.sample_shape,
            #     random_size=False,
            # ),
            # monai.transforms.SpatialPadD(all_keys, spatial_size=self.args.sample_shape),
            monai.transforms.RandFlipD(all_keys, prob=self.args.flip_p, spatial_axis=0),
            monai.transforms.RandFlipD(all_keys, prob=self.args.flip_p, spatial_axis=1),
            monai.transforms.RandFlipD(all_keys, prob=self.args.flip_p, spatial_axis=2),
            # monai.transforms.RandRotate90D(all_keys, prob=self.args.rotate_p),
            monai.transforms.NormalizeIntensityD(img_keys),
            monai.transforms.LambdaD(all_keys, lambda t: t.as_tensor(), track_meta=False),
            monai.transforms.ConcatItemsD(img_keys + [f'{seg_cls}-pred' for seg_cls in self.args.seg_inputs], name=DataKey.IMG),
            # monai.transforms.CopyItemsD(MBDataKey.SUBGROUP_ID, names=DataKey.CLS),
            monai.transforms.ConcatItemsD(seg_keys, name=DataKey.SEG),
            monai.transforms.SelectItemsD([MBDataKey.CASE, DataKey.IMG, DataKey.CLS, DataKey.SEG]),
        ])

    @property
    def val_transform(self):
        img_keys = self.args.input_modalities
        seg_keys = self.args.seg_classes
        pred_keys = [f'{seg_cls}-pred' for seg_cls in self.args.seg_classes]
        bbox_src_key = f'{self.args.crop_ref}-pred'
        all_keys = img_keys + seg_keys + pred_keys
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(all_keys),
            monai.transforms.EnsureChannelFirstD(all_keys),
            monai.transforms.OrientationD(all_keys, axcodes='RAS'),
            monai.transforms.SpacingD(img_keys, pixdim=self.args.spacing, mode=GridSampleMode.BILINEAR),
            monai.transforms.SpacingD(seg_keys, pixdim=self.args.spacing, mode=GridSampleMode.NEAREST),
            monai.transforms.ResizeWithPadOrCropD(all_keys, spatial_size=self.args.pad_crop_size),  # adjust affine
            CropBBoxCenterD(all_keys, bbox_src_key, crop_size=self.args.sample_shape),
            # monai.transforms.CropForegroundD(all_keys, source_key=f'{SegClass.ST}-pred'),
            # monai.transforms.MaskIntensityD(all_keys, mask_key=f'{SegClass.ST}-pred'),
            # monai.transforms.CenterSpatialCropD(
            #     all_keys,
            #     roi_size=self.args.sample_shape,
            # ),
            # monai.transforms.SpatialPadD(all_keys, spatial_size=self.args.sample_shape),
            monai.transforms.NormalizeIntensityD(img_keys),
            monai.transforms.LambdaD(all_keys, MetaTensor.as_tensor, track_meta=False),
            monai.transforms.ConcatItemsD(img_keys + [f'{seg_cls}-pred' for seg_cls in self.args.seg_inputs], name=DataKey.IMG),
            # monai.transforms.CopyItemsD(MBDataKey.SUBGROUP_ID, names=DataKey.CLS),
            monai.transforms.ConcatItemsD(seg_keys, name=DataKey.SEG),
            # monai.transforms.SelectItemsD([MBDataKey.CASE, DataKey.IMG, DataKey.CLS, DataKey.SEG]),
        ])

    def val_dataloader(self, *, include_test: bool = True):
        val_dl = super().val_dataloader()
        if not include_test:
            return val_dl
        return [
            val_dl,
            self.build_eval_dataloader(CacheDataset(
                self.test_data(),
                transform=self.test_transform,
                cache_num=self.args.val_cache_num,
                num_workers=self.args.dataloader_num_workers,
            ))
        ]

    @property
    def test_transform(self):
        return self.val_transform

def parse_age(age: str) -> float:
    if pd.isna(age):
        return math.nan

    match age[-1].lower():
        case 'y':
            return float(age[:-1])
        case 'm':
            return float(age[:-1]) / 12
        case _:
            raise ValueError
