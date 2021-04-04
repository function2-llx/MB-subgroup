from typing import Union, Any, Mapping, Hashable

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import ToTensor, ToTensord, MapTransform, Transform, RandomizableTransform, RandRotate90, \
    RandSpatialCrop, RandSpatialCropd, RandRotate90d
from monai.transforms.utility.array import PILImageImage


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


class ConcatItemsAllowSingled(MapTransform):
    """
    Concatenate specified items from data dictionary together on the first dim to construct a big array.
    Expect all the items are numpy array or PyTorch Tensor.

    """

    def __init__(self, keys: KeysCollection, name: str, dim: int = 0, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be concatenated together.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            name: the name corresponding to the key to store the concatenated data.
            dim: on which dimension to concatenate the items, default is 0.
            allow_missing_keys: don't raise exception if key is missing.

        Raises:
            ValueError: When insufficient keys are given (``len(self.keys) < 2``).

        """
        super().__init__(keys, allow_missing_keys)
        # if len(self.keys) < 2:
        #     raise ValueError("Concatenation requires at least 2 keys.")
        self.name = name
        self.dim = dim

    def __call__(self, data):
        """
        Raises:
            TypeError: When items in ``data`` differ in type.
            TypeError: When the item type is not in ``Union[numpy.ndarray, torch.Tensor]``.

        """
        d = dict(data)
        output = []
        data_type = None
        for key in self.key_iterator(d):
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")
            output.append(d[key])
        if data_type == np.ndarray:
            d[self.name] = np.concatenate(output, axis=self.dim)
        elif data_type == torch.Tensor:
            d[self.name] = torch.cat(output, dim=self.dim)
        else:
            raise TypeError(f"Unsupported data type: {data_type}, available options are (numpy.ndarray, torch.Tensor).")
        return d

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
