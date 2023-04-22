from collections.abc import Callable
from functools import cached_property
import itertools
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from monai import transforms as monai_t
from monai.data import CacheDataset, MetaTensor
from monai.utils import GridSampleMode
from luolib.datamodule import SegDataModule, CrossValDataModule, ExpDataModuleBase
from luolib.utils import DataKey, DataSplit

from mbs.conf import MBSegConf
from mbs.utils.enums import MBDataKey, Modality, SUBGROUPS, SegClass
from mbs.transforms import CropBBoxCenterD

DATASET_ROOT = Path(__file__).parent
DATA_DIR = DATASET_ROOT / 'origin'
PROCESSED_DIR = DATASET_ROOT / 'processed'

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
            # DataKey.CLINICAL: clinical_vec,
        })

    return data

def get_classes(data: list[dict]) -> list:
    return [
        x[MBDataKey.SUBGROUP_ID]
        for x in data
    ]

class MBDataModuleBase(CrossValDataModule):
    conf: MBSegConf

    @cached_property
    def cohort(self):
        return load_cohort()

    @cached_property
    def partitions(self):
        ret = [
            self.cohort[str(fold_id)]
            for fold_id in range(self.conf.num_folds)
        ]
        # trick: select training data for fold-i is by deleting the i-th item
        if self.conf.include_adults:
            ret.append(self.cohort[DataSplit.TRAIN])
        return ret

    def test_data(self) -> Sequence:
        return self.cohort[DataSplit.TEST]

    def all_data(self) -> Sequence:
        return list(itertools.chain(*self.cohort.values()))

class MBSegDataModule(SegDataModule, MBDataModuleBase):
    conf: MBSegConf

    @property
    def train_transform(self) -> Callable:
        img_keys = self.conf.input_modalities
        seg_keys = self.conf.seg_classes
        all_keys = img_keys + seg_keys
        return monai_t.Compose([
            monai_t.LoadImageD(all_keys),
            monai_t.EnsureChannelFirstD(all_keys),
            monai_t.OrientationD(all_keys, axcodes='RAS'),
            monai_t.SpacingD(img_keys, pixdim=self.conf.spacing, mode=GridSampleMode.BILINEAR),
            monai_t.SpacingD(seg_keys, pixdim=self.conf.spacing, mode=GridSampleMode.NEAREST),
            monai_t.ResizeWithPadOrCropD(all_keys, spatial_size=self.conf.pad_crop_size),
            monai_t.RandCropByPosNegLabelD(
                all_keys,
                label_key=SegClass.ST,
                spatial_size=self.conf.sample_shape,
                pos=self.conf.crop_pos,
                neg=self.conf.crop_neg,
                num_samples=self.conf.num_crop_samples,
            ),
            monai_t.RandFlipD(all_keys, prob=self.conf.flip_p, spatial_axis=0),
            monai_t.RandFlipD(all_keys, prob=self.conf.flip_p, spatial_axis=1),
            monai_t.RandFlipD(all_keys, prob=self.conf.flip_p, spatial_axis=2),
            monai_t.RandRotate90D(all_keys, prob=self.conf.rotate_p),
            monai_t.NormalizeIntensityD(img_keys),
            monai_t.LambdaD(all_keys, lambda t: t.as_tensor(), track_meta=False),
            monai_t.ConcatItemsD(img_keys, name=DataKey.IMG),
            monai_t.ConcatItemsD(seg_keys, name=DataKey.SEG),
            monai_t.SelectItemsD([DataKey.IMG, DataKey.SEG]),
        ])

    @property
    def val_transform(self) -> Callable:
        img_keys = self.conf.input_modalities
        seg_keys = self.conf.seg_classes
        all_keys = img_keys + seg_keys
        return monai_t.Compose([
            monai_t.LoadImageD(all_keys),
            monai_t.EnsureChannelFirstD(all_keys),
            monai_t.OrientationD(all_keys, axcodes='RAS'),
            monai_t.SpacingD(img_keys, pixdim=self.conf.spacing, mode=GridSampleMode.BILINEAR),
            monai_t.SpacingD(seg_keys, pixdim=self.conf.spacing, mode=GridSampleMode.NEAREST),
            monai_t.ResizeWithPadOrCropD(all_keys, spatial_size=self.conf.pad_crop_size),
            monai_t.NormalizeIntensityD(img_keys),
            monai_t.LambdaD(all_keys, lambda t: t.as_tensor(), track_meta=False),
            monai_t.ConcatItemsD(img_keys, name=DataKey.IMG),
            monai_t.ConcatItemsD(seg_keys, name=DataKey.SEG),
            monai_t.SelectItemsD([DataKey.IMG, DataKey.SEG, MBDataKey.CASE]),
        ])

    @property
    def test_transform(self):
        return self.val_transform

# class MBDataModule(MBDataModuleBase):
#     args: MBArgs
#
#     def __init__(self, args: MBArgs):
#         super().__init__(args)
#         suffix = f'th{args.th}'
#         if args.do_post:
#             suffix += '-post'
#         for x in itertools.chain(*self.cohort.values()):
#             seg_pred_dir = args.seg_pred_dir / x[MBDataKey.CASE] / suffix
#             for seg_cls in args.seg_classes:
#                 x[f'{seg_cls}-pred'] = seg_pred_dir / f'{seg_cls}.nii.gz'
#                 # x[f'{seg_cls}-pred'] = DATA_DIR / 'image' / x[MBDataKey.CASE] / f'{seg_cls}.nii'
#
#         if args.cls_weights is None:
#             cls_cnt = np.zeros(args.num_cls_classes, dtype=np.float64)
#             for i in range(args.num_folds):
#                 for x in self.cohort[str(i)]:
#                     cls_cnt[x[DataKey.CLS]] += 1
#             args.cls_weights = (1. / cls_cnt).tolist()
#
#     @cached_property
#     def cohort(self):
#         cohort = super().cohort
#         for fold in cohort.values():
#             for x in fold:
#                 subgroup = SUBGROUPS[x[MBDataKey.SUBGROUP_ID]]
#                 x[DataKey.CLS] = self.conf.cls_map[subgroup]
#                 if not self.conf.use_clinical:
#                     x.pop(DataKey.CLINICAL)
#         return {
#             k: list(filter(lambda x: x[DataKey.CLS] != -1, fold))
#             for k, fold in cohort.items()
#         }
#
#     @property
#     def train_transform(self):
#         img_keys = self.conf.input_modalities
#         seg_keys = self.conf.seg_classes
#         pred_keys = [f'{seg_cls}-pred' for seg_cls in self.conf.seg_classes]
#         bbox_src_key = f'{self.conf.crop_ref}-pred'
#         all_keys = img_keys + seg_keys + pred_keys
#         return monai_t.Compose([
#             monai_t.LoadImageD(all_keys),
#             monai_t.EnsureChannelFirstD(all_keys),
#             monai_t.OrientationD(all_keys, axcodes='RAS'),
#             monai_t.SpacingD(img_keys, pixdim=self.conf.spacing, mode=GridSampleMode.BILINEAR),
#             monai_t.SpacingD(seg_keys, pixdim=self.conf.spacing, mode=GridSampleMode.NEAREST),
#             monai_t.ResizeWithPadOrCropD(all_keys, spatial_size=self.conf.pad_crop_size),
#             CropBBoxCenterD(all_keys, bbox_src_key, crop_size=self.conf.sample_shape),
#             # monai_t.CropForegroundD(all_keys, source_key=f'{SegClass.ST}-pred'),
#             # monai_t.MaskIntensityD(all_keys, mask_key=f'{SegClass.ST}-pred'),
#             # monai_t.RandSpatialCropD(
#             #     all_keys,
#             #     roi_size=self.conf.sample_shape,
#             #     random_size=False,
#             # ),
#             # monai_t.SpatialPadD(all_keys, spatial_size=self.conf.sample_shape),
#             monai_t.RandFlipD(all_keys, prob=self.conf.flip_p, spatial_axis=0),
#             monai_t.RandFlipD(all_keys, prob=self.conf.flip_p, spatial_axis=1),
#             monai_t.RandFlipD(all_keys, prob=self.conf.flip_p, spatial_axis=2),
#             # monai_t.RandRotate90D(all_keys, prob=self.conf.rotate_p),
#             monai_t.NormalizeIntensityD(img_keys),
#             monai_t.LambdaD(all_keys, lambda t: t.as_tensor(), track_meta=False),
#             monai_t.ConcatItemsD(img_keys + [f'{seg_cls}-pred' for seg_cls in self.conf.seg_inputs], name=DataKey.IMG),
#             # monai_t.CopyItemsD(MBDataKey.SUBGROUP_ID, names=DataKey.CLS),
#             monai_t.ConcatItemsD(seg_keys, name=DataKey.SEG),
#             monai_t.SelectItemsD([MBDataKey.CASE, DataKey.IMG, DataKey.CLS, DataKey.SEG]),
#         ])
#
#     @property
#     def val_transform(self):
#         img_keys = self.conf.input_modalities
#         seg_keys = self.conf.seg_classes
#         pred_keys = [f'{seg_cls}-pred' for seg_cls in self.conf.seg_classes]
#         bbox_src_key = f'{self.conf.crop_ref}-pred'
#         all_keys = img_keys + seg_keys + pred_keys
#         return monai_t.Compose([
#             monai_t.LoadImageD(all_keys),
#             monai_t.EnsureChannelFirstD(all_keys),
#             monai_t.OrientationD(all_keys, axcodes='RAS'),
#             monai_t.SpacingD(img_keys, pixdim=self.conf.spacing, mode=GridSampleMode.BILINEAR),
#             monai_t.SpacingD(seg_keys, pixdim=self.conf.spacing, mode=GridSampleMode.NEAREST),
#             monai_t.ResizeWithPadOrCropD(all_keys, spatial_size=self.conf.pad_crop_size),  # adjust affine
#             CropBBoxCenterD(all_keys, bbox_src_key, crop_size=self.conf.sample_shape),
#             # monai_t.CropForegroundD(all_keys, source_key=f'{SegClass.ST}-pred'),
#             # monai_t.MaskIntensityD(all_keys, mask_key=f'{SegClass.ST}-pred'),
#             # monai_t.CenterSpatialCropD(
#             #     all_keys,
#             #     roi_size=self.conf.sample_shape,
#             # ),
#             # monai_t.SpatialPadD(all_keys, spatial_size=self.conf.sample_shape),
#             monai_t.NormalizeIntensityD(img_keys),
#             monai_t.LambdaD(all_keys, MetaTensor.as_tensor, track_meta=False),
#             monai_t.ConcatItemsD(img_keys + [f'{seg_cls}-pred' for seg_cls in self.conf.seg_inputs], name=DataKey.IMG),
#             # monai_t.CopyItemsD(MBDataKey.SUBGROUP_ID, names=DataKey.CLS),
#             monai_t.ConcatItemsD(seg_keys, name=DataKey.SEG),
#             # monai_t.SelectItemsD([MBDataKey.CASE, DataKey.IMG, DataKey.CLS, DataKey.SEG]),
#         ])
#
#     def val_dataloader(self, *, include_test: bool = True):
#         val_dl = super().val_dataloader()
#         if not include_test:
#             return val_dl
#         return [
#             val_dl,
#             self.build_eval_dataloader(CacheDataset(
#                 self.test_data(),
#                 transform=self.test_transform,
#                 cache_num=self.conf.val_cache_num,
#                 num_workers=self.conf.dataloader_num_workers,
#             ))
#         ]
#
#     @property
#     def test_transform(self):
#         return self.val_transform

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

SEG_REF = {
    SegClass.AT: Modality.T2,
    SegClass.CT: Modality.T1C,
    SegClass.ST: Modality.T2,
}

def load_merged_plan():
    plan = pd.read_excel(PROCESSED_DIR / 'plan.xlsx', sheet_name='merge', dtype={MBDataKey.NUMBER: 'string'})
    plan.set_index(MBDataKey.NUMBER, inplace=True)
    assert plan.index.unique().size == plan.index.size
    return plan
