import json
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union, Dict, Mapping, Hashable

import numpy as np
import torch
from monai.config import KeysCollection
from monai.data import CacheDataset
from monai.transforms import *
from monai.transforms.utility.array import PILImageImage
from tqdm import tqdm

from utils.dicom_utils import ScanProtocol


class MultimodalDataset(CacheDataset):
    def __init__(self, data: List[Dict], transform: Transform, num_classes: int):
        super().__init__(data, transform)
        self.num_classes = num_classes
        self.labels = [data['label'] for data in self.data]

    def get_weight(self, strategy) -> torch.Tensor:
        if strategy == 'equal':
            return torch.ones(self.num_classes)
        elif strategy == 'invsqrt':
            weight = torch.zeros(self.num_classes)
            for label in self.labels:
                weight[label] += 1
            weight.sqrt_()
            weight = weight.sum() / weight
            weight = weight / weight.sum()
            return weight
        elif strategy == 'inv':
            weight = torch.zeros(self.num_classes)
            for label in self.labels:
                weight[label] += 1
            # weight.sqrt_()
            weight = weight.sum() / weight
            weight = weight / weight.sum()
            return weight
        else:
            raise ValueError(f'unsupported weight strategy of {strategy}')

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

_folds = None
_args = None

def load_folds(args):
    global _folds
    if _folds is None:
        _args = deepcopy(args)
        modalities = list(ScanProtocol)
        loader = Compose([
            LoadImaged(modalities),
            NormalizeIntensityd(modalities, nonzero=False),
            AddChanneld(modalities),
            Orientationd(modalities, axcodes='PLI'),
        ])
        folds_raw = json.load(open('folds.json'))
        _folds = []

        with tqdm(total=sum(len(fold) for fold in folds_raw), ncols=80, desc='loading folds') as bar:
            for fold_raw in folds_raw:
                fold = []
                for info in fold_raw:
                    label = args.target_dict.get(info['subgroup'], None)
                    if label is not None:
                        # T1's time of echo is less than T2's, see [ref](https://radiopaedia.org/articles/t1-weighted-image)
                        scans = sorted(info['scans'], key=lambda scan: json.load(open(Path(scan.replace('.nii.gz', '.json'))))['EchoTime'])
                        assert len(scans) >= 3
                        scans = {
                            protocol: scans[i]
                            for i, protocol in enumerate(ScanProtocol)
                        }
                        data = loader(scans)
                        data['label'] = label
                        fold.append(data)
                    bar.update()
                _folds.append(fold)

    return deepcopy(_folds)

if __name__ == '__main__':
    from run_3d import get_args
    args = get_args()
    load_folds(args)
