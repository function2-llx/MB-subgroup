import itertools
import json
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from monai.data import CacheDataset
from monai.transforms import *
from monai.transforms.utility.array import PILImageImage
from tqdm import tqdm


class MultimodalDataset(CacheDataset):
    def __init__(self, imgs: List[List[np.ndarray]], labels: List[int], transform: Transform, num_classes: int):
        super().__init__(imgs, transform)
        self.labels = labels
        self.num_classes = num_classes

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        imgs: List[torch.Tensor] = super().__getitem__(index)
        return torch.cat(imgs), self.labels[index]

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

def prepare_data_3d(folds, val_id, args):
    train_folds = list(itertools.chain(*[fold for id, fold in enumerate(folds) if id != val_id]))
    val_fold = folds[val_id]

    train_transforms: List[Transform] = []
    if args.aug == 'weak':
        train_transforms = [
            RandFlip(prob=0.2, spatial_axis=0),
            RandRotate90(prob=0.2),
        ]
    elif args.aug == 'strong':
        raise NotImplementedError

    train_transforms: Transform = Compose(train_transforms + [
        Resize((args.sample_size, args.sample_size, args.sample_slices)),
        ToTensorDevice(args.device)
    ])
    val_transforms = Compose([
        Resize(spatial_size=(args.sample_size, args.sample_size, args.sample_slices), mode='area'),
        ToTensorDevice(args.device),
    ])

    return {
        'train': MultimodalDataset(*zip(*train_folds), train_transforms, 4),
        'val': MultimodalDataset(*zip(*val_fold), val_transforms, 4)
    }

folds = None

def get_folds():
    global folds
    if folds is None:
        img_keys = 'img',
        transforms = Compose([
            LoadImaged(img_keys),
            NormalizeIntensityd(img_keys, nonzero=False),
            AddChanneld(img_keys),
            # ScaleIntensityd(img_keys),
            # Spacingd(img_keys, pixdim=(1, 1, 1), mode='bilinear'),
            Orientationd(img_keys, axcodes='PLI'),
        ])
        folds_raw = json.load(open('folds.json'))
        folds = []
        with tqdm(total=sum(len(fold) for fold in folds_raw), ncols=80, desc='loading and normalizing data') as bar:
            for fold_raw in folds_raw:
                fold = []
                for info in fold_raw:
                    # T1's time of echo is less than T2's, see [ref](https://radiopaedia.org/articles/t1-weighted-image)
                    scans = sorted(info['scans'],
                                   key=lambda scan: json.load(open(Path(scan.replace('.nii.gz', '.json'))))['EchoTime'])
                    label = {
                        'WNT': 0,
                        'SHH': 1,
                        'G3': 2,
                        'G4': 3,
                    }[info['subgroup']]
                    imgs = []
                    for scan in scans[:3]:
                        img = transforms({'img': scan})['img']
                        imgs.append(img)
                    fold.append((
                        imgs,
                        label,
                    ))
                    bar.update()
                folds.append(fold)

    return folds


if __name__ == '__main__':
    get_folds()
