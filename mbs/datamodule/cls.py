from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Sampler

from luolib.datamodule.base import DataLoaderConf
from luolib.types import spatial_param_t, tuple2_t, tuple3_t
from luolib import transforms as lt
from monai import transforms as mt
from monai.config import KeysCollection
from monai.data import CacheDataset, DataLoader
from monai.utils import GridSampleMode, convert_to_tensor

from mbs.utils.enums import MBDataKey, SUBGROUPS
from .base import MBDataModuleBase, TransConfBase

__all__ = [
    'MBClsDataModule',
]

@dataclass(kw_only=True)
class ClsTransConf(TransConfBase):
    @dataclass
    class Scale(TransConfBase.Scale):
        prob: float = 0.3
        range: tuple2_t[float] = (0.7, 1.4)
        ignore_dim: int | None = 0

    scale: Scale

    @dataclass
    class Rotate(TransConfBase.Rotate):
        prob: float = 0.4
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
        self.mask_key = mask_key

    def __call__(self, data: Mapping):
        data = dict(data)
        case = data[self.case_key]
        mask_path = self.data_dir / case / 'prob.pt'
        data[self.mask_key] = torch.load(mask_path, 'cpu')
        return data

class STCenterCropD(mt.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        roi_size: spatial_param_t[int],
        mask_key: Hashable,
        th: float = 0.4,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            th: threshold for obtaining ST mask from probability
        """
        super().__init__(keys, allow_missing_keys)
        self.roi_size = roi_size
        self.mask_key = mask_key
        self.th = th
        self.klcc = mt.KeepLargestConnectedComponent(is_onehot=True)
        self.bbox = mt.BoundingRect()

    def __call__(self, data: dict):
        ref = data[self.mask_key][0:1] > self.th
        ref = self.klcc(ref)
        bbox = self.bbox(ref).reshape(-1, 2)
        center = bbox.sum(axis=-1) >> 1
        cropper = mt.SpatialCropD(self.keys, center, self.roi_size, allow_missing_keys=self.allow_missing_keys)
        return cropper(data)

class InputTransformD(mt.Transform):
    def __init__(self, as_tensor: bool = True):
        self.as_tensor = as_tensor

    def __call__(self, data):
        data = dict(data)
        img, mask, label = data['img'], data['mask'], data[MBDataKey.SUBGROUP]
        if self.as_tensor:
            img = convert_to_tensor(img)
        return img, convert_to_tensor(mask), SUBGROUPS.index(label)

class MBSampler(Sampler):
    def __init__(self, num_samples: int, labels: Sequence[int], cls_ratio: float = 0.5, cls_weight: Sequence[float] | None = None):
        """
        Args:
            labels: labels for all samples in dataset
            cls_ratio: how often samples are generated from a class sampled from `cls_weight`
        """
        super().__init__()
        self.num_samples = num_samples
        self.cls_ratio = cls_ratio
        if cls_weight is None:
            cls_weight = [1] * len(SUBGROUPS)
        p = np.array(cls_weight)
        self.p = p / p.sum()
        self.cls_indexes = [[] for _ in range(len(SUBGROUPS))]
        for i, cls in enumerate(labels):
            self.cls_indexes[cls].append(i)
        self.n = len(labels)
        self.cur_uniform = self.n

    def __iter__(self):
        for _ in range(self.num_samples):
            if np.random.rand() < self.cls_ratio:
                cls = int(np.random.choice(np.arange(len(SUBGROUPS)), p=self.p))
                cls_index = self.cls_indexes[cls]
                ret = cls_index[int(np.random.randint(len(cls_index)))]
            else:
                if self.cur_uniform == self.n:
                    self.perm = np.random.permutation(self.n)
                    self.cur_uniform = 0
                ret = int(self.perm[self.cur_uniform])
            yield ret

    def __len__(self):
        return self.num_samples

class MBClsDataModule(MBDataModuleBase):
    def __init__(
        self,
        *args,
        mask_dir: Path,
        trans: ClsTransConf,
        cls_weights: Sequence[float] | None = None,
        sampler_cls_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_dir = mask_dir
        self.trans_conf = trans
        self.cls_weights = cls_weights
        self.sampler_cls_ratio = sampler_cls_ratio

    def train_transform(self):
        conf = self.trans_conf
        AT_th = 0.3
        return mt.Compose(
            [
                lt.nnUNetLoaderD('case', self.data_dir, seg_key=None),
                MaskLoader(self.mask_dir, 'case', 'mask'),
                mt.LambdaD('mask', lambda mask: mask[1:2] > AT_th, overwrite='AT'),
                # ConvertData(),
                lt.ComputeCropIndicesD(
                    'AT',
                    conf.patch_size,
                    0.95, 0.95,
                    '_indices_for_cls',
                    is_onehot=True,
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

    def train_dataloader(self):
        dataset = self.train_dataset()
        conf = self.dataloader_conf
        labels = [
            SUBGROUPS.index(x[MBDataKey.SUBGROUP])
            for x in dataset.data
        ]
        return DataLoader(
            dataset,
            batch_size=conf.train_batch_size,
            sampler=MBSampler(
                conf.num_batches * conf.train_batch_size,
                labels,
                self.sampler_cls_ratio,
                self.cls_weights,
            ),
            num_workers=conf.num_workers,
            pin_memory=conf.pin_memory,
            prefetch_factor=conf.prefetch_factor,
            persistent_workers=conf.persistent_workers,
            collate_fn=self.get_train_collate_fn(),
        )

    def val_transform(self) -> Callable:
        conf = self.trans_conf
        return mt.Compose([
            lt.nnUNetLoaderD('case', self.data_dir, seg_key=None),
            MaskLoader(self.mask_dir, 'case', 'mask'),
            STCenterCropD(['img', 'mask'], conf.patch_size, 'mask'),
            InputTransformD(),
        ])

    def val_dataloader(self):
        val_data = self.val_data()
        real_val_cache_num = min(self.cache_dataset_conf.val_num // 2, len(val_data))
        return [
            self.build_eval_dataloader(
                CacheDataset(
                    data,
                    transform=self.val_transform(),
                    cache_num=cache_num,
                    num_workers=self.dataloader_conf.num_workers,
                ),
                self.dataloader_conf.val_batch_size,
            )
            for data, cache_num in [
                (val_data, real_val_cache_num),
                (self.test_data(), self.cache_dataset_conf.val_num - real_val_cache_num),
            ]
        ]
