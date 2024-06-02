from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.types import Device

from luolib import transforms as lt
from luolib.types import tuple2_t, tuple3_t
from monai import transforms as mt
from monai.utils import GridSampleMode, convert_to_tensor

from .base import MBDataModuleBase, TransConfBase

__all__ = [
    'MBSegDataModule',
]

class ConvertData(mt.Transform):
    def __call__(self, data: dict):
        data = dict(data)
        sem_seg = data.pop('seg')
        seg = torch.empty(2, *sem_seg.shape[1:], dtype=torch.bool)
        seg[0] = (sem_seg == 1) | (sem_seg == 2)
        seg[1] = sem_seg == 2
        data['seg'] = seg

        class_locations: dict = data.pop('class_locations')
        data['class_locations'] = [
            np.ravel_multi_index(class_locations[k][:, 1:].T, seg.shape[1:])
            for k in [(1, 2), 2]
        ]
        return data

class InputTransformD(mt.Transform):
    def __init__(self, as_tensor: bool = True):
        self.as_tensor = as_tensor

    def __call__(self, data):
        data = dict(data)
        img, label = data['img'], data['seg']
        if self.as_tensor:
            img = convert_to_tensor(img)
        return img, convert_to_tensor(label)

@dataclass(kw_only=True)
class SegTransConf(TransConfBase):
    rand_fg_ratios: tuple2_t[float] = (2, 1)
    ST_AT_ratios: tuple2_t[float] = (1, 1)

class MBSegDataModule(MBDataModuleBase):
    def __init__(
        self,
        *args,
        seg_trans: SegTransConf,  # not setting to None: https://github.com/omni-us/jsonargparse/issues/423
        predict_range: tuple2_t[int | None] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seg_trans_conf = seg_trans
        if seg_trans is not None and seg_trans.device != 'cpu':
            self.dataloader_conf.pin_memory = False
        self.predict_range = predict_range

    def fit_data(self) -> dict[str, dict]:
        return {
            k: v for k, v in super().fit_data().items()
            if (self.data_dir / f'{k}_seg.npy').exists()
        }

    def train_transform(self):
        conf = self.seg_trans_conf
        # largely follows nnU-Net
        return mt.Compose(
            [
                lt.nnUNetLoaderD('case', self.data_dir),
                ConvertData(),
                lt.ComputeCropIndicesD('seg', conf.patch_size, cache_path_base_key='path_base'),
                mt.RandIdentity(),
                mt.ToDeviceD(['img', 'seg'], conf.device),
                lt.OneOf(
                    [
                        mt.RandSpatialCropD(['img', 'seg'], conf.patch_size, random_center=True, random_size=False),
                        mt.RandCropByLabelClassesD(
                            ['img', 'seg'],
                            'seg',
                            spatial_size=conf.patch_size,
                            ratios=list(conf.ST_AT_ratios),
                            indices_key='seg_indices',
                            # indices_key='class_locations',
                        ),
                    ],
                    weights=conf.rand_fg_ratios,
                ),
                lt.RandAffineWithIsotropicScaleD(
                    ['img', 'seg'],
                    conf.patch_size,
                    conf.scale.prob,
                    scale_range=conf.scale.range,
                    spatial_dims=3,
                    ignore_dim=conf.scale.ignore_dim,
                ),
                mt.RandAffineD(
                    ['img', 'seg'],
                    conf.patch_size,
                    conf.rotate.prob,
                    rotate_range=conf.rotate.range,
                ),
                *[
                    mt.RandFlipD(['img', 'seg'], 0.5, i)
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

    def val_transform(self) -> Callable:
        return mt.Compose(
            [
                lt.nnUNetLoaderD('case', self.data_dir),
                ConvertData(),
                InputTransformD(),
            ],
            lazy=True,
        )

    def predict_transform(self) -> Callable:
        return mt.Compose([
            lt.nnUNetLoaderD('case', self.data_dir, seg_key=None),
            mt.ToTensorD('img', track_meta=False),
        ])

    def predict_data(self):
        data = [
            self.extract_case_data(number)
            for number in self.plan.index
        ]
        if self.predict_range is not None:
            data = data[slice(*self.predict_range)]
        return data
