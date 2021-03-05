import itertools
from typing import List, Tuple

import torch
from monai.data import CacheDataset
from monai.transforms import LoadImage, ScaleIntensity, AddChannel, Resize, RandRotate90, ToTensor, Compose, Transform

class MultimodalDataset(CacheDataset):
    def __init__(self, imgs: List[List[str]], labels: List[int], transform: Transform):
        super().__init__(imgs, transform)
        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        imgs = super().__getitem__(index)
        return torch.cat(imgs), self.labels[index]

def prepare_data_3d(folds, val_id, args):
    train_folds = list(itertools.chain(*[fold for id, fold in enumerate(folds) if id != val_id]))
    val_fold = folds[val_id]

    train_transforms = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        AddChannel(),
        Resize((args.sample_size, args.sample_size, args.sample_slices)),
        RandRotate90(),
        ToTensor(),
    ])
    val_transforms = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        AddChannel(),
        Resize((args.sample_size, args.sample_size, args.sample_slices)),
        ToTensor()
    ])

    return {
        'train': MultimodalDataset(*zip(*train_folds), train_transforms),
        'val': MultimodalDataset(*zip(*val_fold), val_transforms)
    }
