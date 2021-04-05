from typing import Union, Any, Mapping, Hashable, Sequence, Optional, Tuple

import numpy as np
import torch
from monai.config import KeysCollection
import monai.transforms as monai_transforms
from monai.transforms import ToTensor, ToTensord, MapTransform, Transform, RandomizableTransform, RandRotate90, \
    RandSpatialCrop, RandSpatialCropd, RandRotate90d
from monai.transforms.utility.array import PILImageImage
from monai.utils import ensure_tuple_rep, fall_back_tuple


class ToTensorDevice(ToTensor):
    """
    Converts the input image to a tensor without applying any other transformations and to device.
    """

    def __init__(self, device: torch.device = 'cuda'):
        self.device = device

    def __call__(self, img: Union[np.ndarray, torch.Tensor, PILImageImage]) -> torch.Tensor:
        """
        Apply the transform to `img` and make it contiguous and on device.
        """
        return super().__call__(img).to(self.device)


class ToTensorDeviced(ToTensord):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToTensor`.
    """

    def __init__(self, keys: KeysCollection, device: torch.device = 'cuda', allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = ToTensorDevice(device)

class SampleSlices(Transform):
    def __init__(self, start: int, sample_slices: int, spacing: int):
        assert sample_slices > 0 and spacing > 0
        self.start = start
        self.spacing = spacing
        self.sample_slices = sample_slices
        self.end = start + (sample_slices - 1) * spacing + 1

    def __call__(self, data: Union[np.ndarray, torch.Tensor]):
        assert self.end <= data.shape[-1]
        ret = data[..., self.start:self.end:self.spacing]
        assert ret.shape[-1] == self.sample_slices
        return ret

class RandSampleSlicesd(MapTransform, RandomizableTransform):
    def __init__(self, keys: KeysCollection, sample_slices: int, spacing: int):
        super().__init__(keys)
        assert sample_slices > 1
        self.sample_slices = sample_slices
        # both inclusive
        self.spacing_range = (max(1, spacing // 2), spacing * 3 // 2)

        self._start = 0
        self._spacing = 0

    # n_slices: slices of current image
    def randomize(self, img_slices) -> None:
        assert img_slices >= self.spacing_range[0] * (self.sample_slices - 1) + 1
        spacing_max = min(self.spacing_range[1], (img_slices - 1) // (self.sample_slices - 1))
        self._spacing = self.R.randint(self.spacing_range[0], spacing_max + 1)
        self._start = self.R.randint(img_slices - self._spacing * (self.sample_slices - 1))

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        shape = d[self.keys[0]].shape
        assert len(shape) >= 3
        self.randomize(shape[-1])

        sample_slices = SampleSlices(self._start, self.sample_slices, self._spacing)
        for key in self.key_iterator(d):
            d[key] = sample_slices(d[key])

        return d

class RandSpatialCropWithRatiod(RandSpatialCropd):
    def __init__(self, keys: KeysCollection, roi_ratio: Union[Sequence[float], float], random_center: bool = True, random_size: bool = True,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, -1, random_center, random_size, allow_missing_keys)
        self.roi_ratio = roi_ratio

    def randomize(self, img_size: Sequence[int]) -> None:
        ratios = fall_back_tuple(self.roi_ratio, [1] * len(img_size))
        self.roi_size = [int(size * ratio) for size, ratio in zip(img_size, ratios)]
        super().randomize(img_size)
