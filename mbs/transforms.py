from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import torch

import monai
from monai.config import KeysCollection
import monai.transforms
from monai.transforms import MapTransform, RandSpatialCropd, RandomizableTransform, Transform
from monai.utils import Method, fall_back_tuple

class SampleSlices(Transform):
    def __init__(self, start: int, num_slices: int):
        self.start = start
        self.num_slices = num_slices

    def __call__(self, data: np.ndarray | torch.Tensor):
        # assert self.end <= data.shape[-1]
        img_slices = data.shape[-1]
        slices = [self.start + i * img_slices // self.num_slices for i in range(self.num_slices)]
        ret = data[..., slices]
        return ret

# class SampleSlicesd(MapTransform):
#     def __init__(self, keys: KeysCollection, start: int, num_slices: int):
#         super().__init__(keys)
#         self.sample_slices = SampleSlices(start, num_slices)
#
#     def __call__(self, data: Mapping[Hashable, np.ndarray]):
#         d = dict(data)
#         # sample_slices = SampleSlices(self._start, self.num_slices)
#         for key in self.key_iterator(d):
#             d[key] = self.sample_slices(d[key])
#
#         return d
#
# SampleSlicesD = SampleSlicesd

# class RandSampleSlicesd(MapTransform, RandomizableTransform):
#     def __init__(self, keys: KeysCollection, sample_slices: int, spacing: int):
#         super().__init__(keys)
#         assert sample_slices > 1
#         self.sample_slices = sample_slices
#         # both inclusive
#         self.spacing_range = (max(1, spacing // 2), spacing * 3 // 2)
#
#         self._start = 0
#         self._spacing = 0
#
#     # n_slices: slices of current image
#     def randomize(self, img_slices) -> None:
#         assert img_slices >= self.spacing_range[0] * (self.sample_slices - 1) + 1
#         spacing_max = min(self.spacing_range[1], (img_slices - 1) // (self.sample_slices - 1))
#         self._spacing = self.R.randint(self.spacing_range[0], spacing_max + 1)
#         self._start = self.R.randint(img_slices - self._spacing * (self.sample_slices - 1))
#
#     def __call__(self, data: Mapping[Hashable, np.ndarray]):
#         d = dict(data)
#         shape = d[self.keys[0]].shape
#         assert len(shape) >= 3
#         self.randomize(shape[-1])
#
#         sample_slices = SampleSlices(self._start, self.sample_slices, self._spacing)
#         for key in self.key_iterator(d):
#             d[key] = sample_slices(d[key])
#
#         return d

class RandSampleSlicesd(MapTransform, RandomizableTransform):
    def __init__(self, keys: KeysCollection, num_slices: int):
        super().__init__(keys)
        self.num_slices = num_slices
        self._start = 0
        # self.img_slices = img
        # self._sample_spacing = img_slices / sample_slices

    # n_slices: slices of current image
    def randomize(self, img_slices) -> None:
        assert self.num_slices <= img_slices
        self._start = self.R.randint((img_slices - 1) // self.num_slices + 1)

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        shape = d[self.keys[0]].shape
        assert len(shape) >= 3
        self.randomize(shape[-1])

        sample_slices = SampleSlices(self._start, self.num_slices)
        for key in self.key_iterator(d):
            d[key] = sample_slices(d[key])

        return d

RandSampleSlicesD = RandSampleSlicesd

class RandSpatialCropWithRatiod(RandSpatialCropd):
    def __init__(
        self,
        keys: KeysCollection,
        roi_ratio: Sequence[float] | float,
        random_center: bool = True,
        random_size: bool = True,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, -1, random_center, random_size, allow_missing_keys)
        self.roi_ratio = roi_ratio

    def randomize(self, img_size: Sequence[int]):
        ratios = fall_back_tuple(self.roi_ratio, [1] * len(img_size))
        self.roi_size = [int(size * ratio) for size, ratio in zip(img_size, ratios)]
        super().randomize(img_size)

class SquareWithPadOrCrop(monai.transforms.Transform):
    def __init__(self, trans: bool = False):
        self.ref_dim = 0
        self.change_dim = 1
        if trans:
            self.ref_dim, self.change_dim = self.change_dim, self.ref_dim

    def __call__(self, data: np.ndarray) -> np.ndarray:
        size = data.shape[self.ref_dim + 1]
        roi_slices = [slice(None) for _ in range(data.ndim - 1)]
        roi_slices[self.change_dim] = slice(0, size)
        cropper = monai.transforms.SpatialCrop(roi_slices=roi_slices)
        data = cropper(data)
        pad_size = list(data.shape[1:])
        pad_size[self.change_dim] = size
        padder = monai.transforms.SpatialPad(spatial_size=pad_size, method=Method.END)
        return padder(data)

class SquareWithPadOrCropD(monai.transforms.MapTransform):
    backend = SquareWithPadOrCrop.backend

    def __init__(self, keys: KeysCollection, trans: bool = False, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.squarer = SquareWithPadOrCrop(trans)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.squarer(d[key])
        return d

class CreateForegroundMaskD(monai.transforms.MapTransform):
    def __init__(self, keys, mask_key):
        super().__init__(keys)
        self.mask_key = mask_key

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d = dict(data)
        d[self.mask_key] = np.logical_or.reduce([d[key] > 0 for key in self.key_iterator(d)]).astype(np.uint8)
        return d

class CropBBoxCenterD(monai.transforms.MapTransform):
    def __init__(self, keys, bbox_src_key: str, crop_size: Sequence[int]):
        super().__init__(keys)
        self.bbox_src_key = bbox_src_key
        self.bbox = monai.transforms.BoundingRect()
        self.crop_size = crop_size

    def __call__(self, data):
        d = dict(data)
        bbox_src = d[self.bbox_src_key]
        assert bbox_src.shape[0] == 1
        bbox = self.bbox(bbox_src)[0]
        center = [
            bbox[i << 1 | 1] + bbox[i << 1] >> 1
            for i in range(len(bbox) >> 1)
        ]
        cropper = monai.transforms.Compose([
            monai.transforms.SpatialCropD(self.keys, roi_center=center, roi_size=self.crop_size),
            monai.transforms.ResizeWithPadOrCropD(self.keys, spatial_size=self.crop_size),
        ])
        return cropper(d)
