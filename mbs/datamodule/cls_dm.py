from functools import cached_property
import itertools as it

from pytorch_lightning.trainer.states import RunningStage

from luolib.datamodule.cls_dm import ClsDataModule
from luolib.reader import PyTorchReader
from luolib.transforms import SpatialCropWithSpecifiedCenterD, RandAffineCropD, RandAdjustContrastD, RandGammaCorrectionD, SimulateLowResolutionD
from monai import transforms as monai_t

from luolib.utils import DataKey
from monai.data import CacheDataset
from monai.utils import PytorchPadMode, GridSampleMode

from .base import MBDataModuleBase
from ..conf import MBClsConf
from ..utils.enums import SegClass

class MBClsDataModule(ClsDataModule, MBDataModuleBase):
    conf: MBClsConf
    pred_keys = list(map(lambda seg_class: f'{seg_class}-pred', SegClass))

    @cached_property
    def split_cohort(self):
        conf = self.conf
        split_cohort = super().split_cohort
        center = MBClsConf.load_center(conf)
        for split, cohort in split_cohort.items():
            for data in cohort:
                case = data[DataKey.CASE]
                data.update(
                    {
                        f'{seg_class}-pred': MBClsConf.get_pred_path(conf, case, seg_class)
                        for seg_class in SegClass
                    },
                    center=center[case],
                )
        return split_cohort


    def load_data_transform(self, _stage):
        return [
            monai_t.LoadImageD(DataKey.IMG, ensure_channel_first=False, image_only=True),
            monai_t.LoadImageD(self.pred_keys, ensure_channel_first=True, image_only=True, reader=PyTorchReader),
        ]

    def intensity_normalize_transform(self, _stage):
        return []

    def spatial_normalize_transform(self, stage: RunningStage):
        transforms = []
        if stage != RunningStage.TRAINING:
            conf = self.conf
            transforms.extend([
                SpatialCropWithSpecifiedCenterD(
                    [DataKey.IMG, *self.pred_keys],
                    center_key='center',
                    roi_size=conf.sample_shape,
                ),
                monai_t.SpatialPadD(
                    [DataKey.IMG, *self.pred_keys],
                    spatial_size=conf.sample_shape,
                    mode=PytorchPadMode.CONSTANT,
                    pad_min=True,
                ),
            ])
        return transforms

    def aug_transform(self):
        conf = self.conf
        all_keys = [DataKey.IMG, *self.pred_keys]
        return [
            RandAffineCropD(
                all_keys,
                conf.sample_shape,
                [GridSampleMode.BILINEAR, *it.repeat(GridSampleMode.NEAREST, len(self.pred_keys))],
                conf.rotate_range,
                conf.rotate_p,
                conf.scale_range,
                conf.scale_p,
                conf.spatial_dims,
                center_generator=lambda data: data[f'center'],
            ),
            monai_t.RandGaussianNoiseD(
                DataKey.IMG,
                prob=conf.gaussian_noise_p,
                std=conf.gaussian_noise_std,
            ),
            monai_t.RandGaussianSmoothD(
                DataKey.IMG,
                conf.gaussian_smooth_std_range,
                conf.gaussian_smooth_std_range,
                conf.gaussian_smooth_std_range,
                prob=conf.gaussian_smooth_p,
                isotropic_prob=conf.gaussian_smooth_isotropic_prob,
            ),
            monai_t.RandScaleIntensityD(DataKey.IMG, factors=conf.scale_intensity_factor, prob=conf.scale_intensity_p),
            monai_t.RandShiftIntensityD(DataKey.IMG, offsets=conf.shift_intensity_offset, prob=conf.shift_intensity_p),
            RandAdjustContrastD(DataKey.IMG, conf.adjust_contrast_range, conf.adjust_contrast_p),
            SimulateLowResolutionD(DataKey.IMG, conf.simulate_low_res_zoom_range, conf.simulate_low_res_p, conf.dummy_dim),
            RandGammaCorrectionD(DataKey.IMG, conf.gamma_p, conf.gamma_range),
            *[
                monai_t.RandFlipD(all_keys, prob=conf.flip_p, spatial_axis=i)
                for i in range(conf.spatial_dims)
            ],
        ]

    def post_transform(self, _stage):
        return [
            monai_t.SelectItemsD([DataKey.IMG, *self.pred_keys, DataKey.CLS]),
            monai_t.LambdaD([DataKey.IMG, *self.pred_keys], lambda x: x.as_tensor(), track_meta=False)
        ]

    def val_dataloader(self):
        conf = self.conf
        val_data = self.val_data()
        real_val_cache_num = min(conf.val_cache_num // 2, len(val_data))
        return [
            self.build_eval_dataloader(
                CacheDataset(
                    data,
                    transform=self.val_transform(),
                    cache_num=conf.val_cache_num,
                    num_workers=conf.dataloader_num_workers,
                )
            )
            for data, cache_num in [
                (val_data, real_val_cache_num),
                (self.test_data(), conf.val_cache_num - real_val_cache_num),
            ]
        ]
