from typing import Union, Mapping, Hashable, Sequence

import monai.transforms as monai_transforms
import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import ToTensor, ToTensord, MapTransform, Transform, RandomizableTransform, RandSpatialCropd
from monai.transforms.utility.array import PILImageImage
from monai.utils import fall_back_tuple

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
    def __init__(self, start: int, num_slices: int):
        self.start = start
        self.num_slices = num_slices

    def __call__(self, data: Union[np.ndarray, torch.Tensor]):
        # assert self.end <= data.shape[-1]
        img_slices = data.shape[-1]
        slices = [self.start + i * img_slices // self.num_slices for i in range(self.num_slices)]
        ret = data[..., slices]
        return ret

class SampleSlicesd(MapTransform):
    def __init__(self, keys: KeysCollection, start: int, num_slices: int):
        super().__init__(keys)
        self.sample_slices = SampleSlices(start, num_slices)

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        # sample_slices = SampleSlices(self._start, self.num_slices)
        for key in self.key_iterator(d):
            d[key] = self.sample_slices(d[key])

        return d

SampleSlicesD = SampleSlicesd

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
    def __init__(self, keys: KeysCollection, roi_ratio: Union[Sequence[float], float], random_center: bool = True, random_size: bool = True,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, -1, random_center, random_size, allow_missing_keys)
        self.roi_ratio = roi_ratio

    def randomize(self, img_size: Sequence[int]) -> None:
        ratios = fall_back_tuple(self.roi_ratio, [1] * len(img_size))
        self.roi_size = [int(size * ratio) for size, ratio in zip(img_size, ratios)]
        super().randomize(img_size)

class ConvertToMultiChannelBasedOnBratsClassesd(monai_transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
        - the GD-enhancing tumor (ET — label 4)
        - the peritumoral edema (ED — label 2)
        - and the necrotic and non-enhancing tumor core (NCR/NET — label 1)

    # label 1 is the peritumoral edema
    # label 2 is the GD-enhancing tumor
    # label 3 is the necrotic and non-enhancing tumor core
    # The possible classes are TC (Tumor core), WT (Whole tumor)
    # and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge labels 1, 2 and 4 to construct WT
            result.append(d[key] != 0)
            # merge label 1 and label 4 to construct TC
            result.append((d[key] == 1) | (d[key] == 4))
            # label 4 is AT
            result.append(d[key] == 4)
            d[key] = np.concatenate(result, axis=0).astype(np.float32)
        return d
