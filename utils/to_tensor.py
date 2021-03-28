from typing import Union

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import ToTensor, ToTensord
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