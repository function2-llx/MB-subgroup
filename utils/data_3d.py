import itertools
from typing import List, Tuple

import numpy as np
from monai.data import CacheDataset, Dataset
from monai.transforms import (
    LoadImage,
    ScaleIntensity,
    AddChannel,
    Resize,
    RandRotate90,
    ToTensor,
    Compose,
    Transform,
)
import torch

class MultimodalDataset(CacheDataset):
    def __init__(self, imgs: List[List[np.ndarray]], labels: List[int], transform: Transform, num_classes: int):
        super().__init__(imgs, transform)
        self.labels = labels
        self.num_classes = num_classes

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        imgs = super().__getitem__(index)
        return torch.cat(imgs), self.labels[index]

    def get_weight(self, strategy) -> torch.FloatTensor:
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

def prepare_data_3d(folds, val_id, args):
    train_folds = list(itertools.chain(*[fold for id, fold in enumerate(folds) if id != val_id]))
    val_fold = folds[val_id]

    train_transforms = Compose([
        ScaleIntensity(),
        AddChannel(),
        Resize((args.sample_size, args.sample_size, args.sample_slices)),
        RandRotate90(),
        ToTensor(),
    ]).set_random_state(args.seed)
    val_transforms = Compose([
        ScaleIntensity(),
        AddChannel(),
        Resize((args.sample_size, args.sample_size, args.sample_slices)),
        ToTensor()
    ])

    return {
        'train': MultimodalDataset(*zip(*train_folds), train_transforms, 4),
        'val': MultimodalDataset(*zip(*val_fold), val_transforms, 4)
    }
