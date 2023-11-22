from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from luolib.types import tuple2_t, tuple3_t
from luolib import transforms as lt
from monai import transforms as mt
from monai.data import CacheDataset
from monai.utils import GridSampleMode, convert_to_tensor

from .base import MBDataModuleBase, TransConfBase

@dataclass(kw_only=True)
class ClsTransConf(TransConfBase):
    @dataclass
    class Scale(TransConfBase.Scale):
        prob: float = 0.2
        range: tuple2_t[float] = (0.7, 1.4)
        ignore_dim: int | None = 0

    scale: Scale

    @dataclass
    class Rotate(TransConfBase.Rotate):
        prob: float = 0.2
        range: tuple3_t[float] = (np.pi / 2, 0, 0)

    rotate: Rotate

    @dataclass
    class GaussianNoise(TransConfBase.GaussianNoise):
        prob: float = 0.1
        max_std: float = 0.1

    gaussian_noise: GaussianNoise

    @dataclass
    class GaussianSmooth(TransConfBase.GaussianSmooth):
        prob: float = 0.2
        prob_per_channel: float = 0.5
        sigma_x: tuple2_t[float] = (0.5, 1)
        sigma_y: tuple2_t[float] = (0.5, 1)
        sigma_z: tuple2_t[float] = (0.5, 1)

    gaussian_smooth: GaussianSmooth

    @dataclass
    class ScaleIntensity(TransConfBase.ScaleIntensity):
        prob: float = 0.15
        range: tuple2_t[float] = (0.75, 1.25)

        @property
        def factors(self):
            return self.range[0] - 1, self.range[1] - 1

    scale_intensity: ScaleIntensity

    @dataclass
    class AdjustContrast(TransConfBase.AdjustContrast):
        prob: float = 0.15
        range: tuple2_t[float] = (0.75, 1.25)
        preserve_intensity_range: bool = True

    adjust_contrast: AdjustContrast

    @dataclass
    class SimulateLowResolution(TransConfBase.SimulateLowResolution):
        prob: float = 0.25
        prob_per_channel: float = 0.5
        zoom_range: tuple2_t[float] = (0.5, 1)
        downsample_mode: str | int = GridSampleMode.NEAREST
        upsample_mode: str | int = GridSampleMode.BICUBIC

    simulate_low_resolution: SimulateLowResolution

    @dataclass
    class GammaCorrection(TransConfBase.GammaCorrection):
        prob: float = 0.3
        range: tuple2_t[float] = (0.7, 1.5)
        prob_invert: float = 0.75
        retain_stats: bool = True

    gamma_correction: GammaCorrection

class MaskLoader(mt.Transform):
    def __init__(self, data_dir: Path, case_key: Hashable, mask_key: Hashable):
        self.data_dir = data_dir
        self.case_key = case_key
        self.pred_key = mask_key

    def __call__(self, data: Mapping):
        data = dict(data)
        case = data[self.case_key]
        mask_path = self.data_dir / case / 'prob.pt'
        return torch.load(mask_path, 'cpu')

class InputTransformD(mt.Transform):
    def __init__(self, as_tensor: bool = True):
        self.as_tensor = as_tensor

    def __call__(self, data):
        data = dict(data)
        img, mask = data['img'], data['mask']
        if self.as_tensor:
            img = convert_to_tensor(img)
        return img, convert_to_tensor(mask)

class MBClsDataModule(MBDataModuleBase):
    def __init__(
        self,
        *args,
        mask_dir: Path,
        trans: ClsTransConf,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_dir = mask_dir
        self.trans_conf = trans

    # @property
    # def pred_keys(self):
    #     return self.conf.pred_keys

    # @cached_property
    # def split_cohort(self):
    #     conf = self.conf
    #     split_cohort = super().split_cohort
    #     center = MBClsConf.load_center(conf)
    #     cls_map = get_cls_map(conf.cls_scheme)
    #     for split, cohort in split_cohort.items():
    #         cohort = list(filter(lambda x: cls_map[x[MBDataKey.SUBGROUP]] != -1, cohort))
    #         for data in cohort:
    #             case = data[DataKey.CASE]
    #             data[DataKey.CLS] = cls_map[data[MBDataKey.SUBGROUP]]
    #             data.update(
    #                 **{
    #                     f'{seg_class}-pred': MBClsConf.get_pred_path(conf, case, seg_class)
    #                     for seg_class in SegClass
    #                 },
    #                 **{
    #                     DataKey.MASK: conf.seg_pred_dir / 'pred' / case / 'seg-prob.pt'
    #                 },
    #                 center=center[case],
    #             )
    #         split_cohort[split] = cohort
    #     return split_cohort

    def train_transform(self):
        conf = self.trans_conf
        return mt.Compose(
            [
                lt.nnUNetLoaderD('case', self.data_dir, seg_key=None),
                MaskLoader(self.mask_dir, 'case', 'mask'),
                mt.LambdaD('mask', lambda mask: mask[1:2], overwrite='AT'),  # get mask of tumor
                # ConvertData(),
                lt.ComputeCropIndicesD(
                    'AT',
                    conf.patch_size,
                    0.9, 0.9,
                    '_indices_for_cls',
                    cache_path_base_key='path_base',
                ),
                mt.RandIdentity(),
                mt.ToDeviceD(['img', 'mask'], conf.device),
                mt.RandCropByLabelClassesD(
                    ['img', 'mask'],
                    'mask',
                    spatial_size=conf.patch_size,
                    ratios=[1],
                    indices_key='AT_indices_for_cls',
                ),
                lt.RandAffineWithIsotropicScaleD(
                    ['img', 'mask'],
                    conf.patch_size,
                    conf.scale.prob,
                    scale_range=conf.scale.range,
                    spatial_dims=3,
                    ignore_dim=conf.scale.ignore_dim,
                ),
                mt.RandAffineD(
                    ['img', 'mask'],
                    conf.patch_size,
                    conf.rotate.prob,
                    rotate_range=conf.rotate.range,
                ),
                *[
                    mt.RandFlipD(['img', 'mask'], 0.5, i)
                    for i in range(3)
                ],
                mt.RandGaussianNoiseD('img', conf.gaussian_noise.prob, std=conf.gaussian_noise.max_std),
                lt.RandDictWrapper(
                    'img',
                    lt.RandGaussianSmooth(
                        conf.gaussian_smooth.sigma_x,
                        conf.gaussian_smooth.sigma_y,
                        conf.gaussian_smooth.sigma_z,
                        conf.gaussian_smooth.prob,
                        prob_per_channel=conf.gaussian_smooth.prob_per_channel,
                    ),
                ),
                mt.RandScaleIntensityD(
                    'img',
                    conf.scale_intensity.factors,
                    conf.scale_intensity.prob,
                    True,
                ),
                lt.RandDictWrapper(
                    'img',
                    lt.RandAdjustContrast(
                        conf.adjust_contrast.prob,
                        conf.adjust_contrast.range,
                        conf.adjust_contrast.preserve_intensity_range,
                    ),
                ),
                lt.RandDictWrapper(
                    'img',
                    lt.RandSimulateLowResolution(
                        conf.simulate_low_resolution.prob,
                        conf.simulate_low_resolution.prob_per_channel,
                        conf.simulate_low_resolution.downsample_mode,
                        conf.simulate_low_resolution.upsample_mode,
                        conf.simulate_low_resolution.zoom_range,
                    ),
                ),
                lt.RandDictWrapper(
                    'img',
                    lt.RandGammaCorrection(
                        conf.gamma_correction.prob,
                        conf.gamma_correction.range,
                        conf.gamma_correction.prob_invert,
                        conf.gamma_correction.retain_stats,
                    ),
                ),
                InputTransformD(),
            ],
            lazy=True,
            overrides={
                'img': {'mode': GridSampleMode.BICUBIC},
                'seg': {'mode': GridSampleMode.BILINEAR},
            }
        )

    def val_dataloader(self):
        conf = self.conf
        val_data = self.val_data()
        if conf.val_test:
            real_val_cache_num = min(conf.val_cache_num // 2, len(val_data))
            return [
                self.build_eval_dataloader(
                    CacheDataset(
                        data,
                        transform=self.val_transform(),
                        cache_num=cache_num,
                        num_workers=conf.dataloader_num_workers,
                    )
                )
                for data, cache_num in [
                    (val_data, real_val_cache_num),
                    (self.test_data(), conf.val_cache_num - real_val_cache_num),
                ]
            ]
        else:
            return self.build_eval_dataloader(
                CacheDataset(
                    val_data,
                    transform=self.val_transform(),
                    cache_num=conf.val_cache_num,
                    num_workers=conf.dataloader_num_workers,
                )
            )

# class MBM2FClsDataModule(MBClsDataModule):
#     def load_data_transform(self, _stage):
#         return [
#             mt.LoadImageD(DataKey.IMG, ensure_channel_first=False, image_only=True),
#             mt.LoadImageD(self.pred_keys, ensure_channel_first=False, image_only=True, reader=PyTorchReader),
#         ]
