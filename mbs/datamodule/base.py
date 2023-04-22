from functools import cached_property
import itertools
import math
from typing import Sequence

import pandas as pd

from luolib.datamodule import CrossValDataModule
from luolib.utils import DataSplit

from mbs.conf import MBConfBase
from mbs.utils.enums import CLINICAL_DIR, MBDataKey, PROCESSED_DIR

def load_clinical():
    clinical = pd.read_excel(CLINICAL_DIR / 'clinical-com.xlsx', dtype='string').set_index('住院号')
    return clinical

# def load_split_cohort(data_dir: Path, requires_seg: bool):
#     plan =  load_merged_plan()
#     clinical = load_clinical()
#     data = {}
#     for number, info in plan.iterrows():
#         sex: str = clinical.loc[number, 'sex']
#         sex_vec = torch.zeros(2)
#         if sex.lower() == 'f':
#             sex_vec[0] = 1
#         if sex.lower() == 'm':
#             sex_vec[1] = 1
#         age = clinical.loc[number, 'age']
#         age_vec = torch.zeros(2)
#         if not pd.isna(age):
#             age_vec[0] = 1
#             age_vec[1] = parse_age(age) / 100
#         clinical_vec = torch.cat([sex_vec, age_vec])
#
#         case_data_dir = data_dir / number
#         data.setdefault(str(info['split']), []).append({
#             MBDataKey.NUMBER: number,
#             MBDataKey.SUBGROUP: info['subgroup'],
#             MBDataKey.SUBGROUP_ID: SUBGROUPS.index(info['subgroup']),
#             **{
#                 key: img_path
#                 for key in [DataKey.IMG, DataKey.SEG] if (img_path := case_data_dir / f'{key}.npy').exists()
#             },
#             MBDataKey.CLINICAL: clinical_vec,
#         })
#
#     return data

# def get_classes(data: list[dict]) -> list:
#     return [
#         x[MBDataKey.SUBGROUP_ID]
#         for x in data
#     ]

class MBDataModuleBase(CrossValDataModule):
    conf: MBConfBase

    @cached_property
    def split_cohort(self) -> dict[str, Sequence]:
        raise NotImplementedError

    @cached_property
    def partitions(self):
        ret = [
            self.split_cohort[str(fold_id)]
            for fold_id in range(self.conf.num_folds)
        ]
        # trick: select training data for fold-i is by deleting the i-th item
        if self.conf.include_adults:
            ret.append(self.split_cohort[DataSplit.TRAIN])
        return ret

    def test_data(self) -> Sequence:
        return self.split_cohort[DataSplit.TEST]

    def all_data(self) -> Sequence:
        return list(itertools.chain(*self.split_cohort.values()))

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

def load_merged_plan():
    plan = pd.read_excel(PROCESSED_DIR / 'plan.xlsx', sheet_name='merge', dtype={MBDataKey.NUMBER: 'string'})
    plan.set_index(MBDataKey.NUMBER, inplace=True)
    assert plan.index.unique().size == plan.index.size
    return plan

def load_split() -> pd.Series:
    split = pd.read_excel(PROCESSED_DIR / 'split.xlsx', dtype=str)
    split.set_index(MBDataKey.NUMBER, inplace=True)
    return split['split']
