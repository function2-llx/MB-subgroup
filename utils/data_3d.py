import itertools
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from monai.data import CacheDataset
from monai.transforms import *
from tqdm import tqdm


class MultimodalDataset(CacheDataset):
    def __init__(self, imgs: List[List[np.ndarray]], labels: List[int], transform: Transform, num_classes: int):
        super().__init__(imgs, transform)
        self.labels = labels
        self.num_classes = num_classes

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        imgs = super().__getitem__(index)
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

def prepare_data_3d(folds, val_id, args):
    train_folds = list(itertools.chain(*[fold for id, fold in enumerate(folds) if id != val_id]))
    val_fold = folds[val_id]

    train_transforms = Compose([
        RandRotate90(),
        ToTensor(),
    ]).set_random_state(args.seed)
    val_transforms = Compose([
        ToTensor(),
    ])

    return {
        'train': MultimodalDataset(*zip(*train_folds), train_transforms, 4),
        'val': MultimodalDataset(*zip(*val_fold), val_transforms, 4)
    }

folds = None

def get_folds(args):
    global folds
    spacing = Spacing((1, 1, 2))
    spacing(np.random.randn(5, 5, 5))
    if folds is None:
        transforms = Compose([
            LoadImaged(keys=('img',)),
            ScaleIntensityd(keys=('img',)),
            Spacingd(keys=('img',), pixdim=(1.5, 1.5, 5.0), mode='bilinear'),
            Orientationd(keys=('img',), axcodes='PLI'),
            AddChanneld(keys=('img',)),
        ])
        folds_raw = json.load(open('folds.json'))
        folds = []
        with tqdm(total=sum(len(fold) for fold in folds_raw), ncols=80, desc='loading and normalizing data') as bar:
            for fold_raw in folds_raw:
                fold = []
                for info in fold_raw:
                    scans = sorted(info['scans'], key=lambda scan: json.load(open(Path(scan.replace('.nii.gz', '.json'))))['EchoTime'])
                    print(scans)
                    img = torch.cat([transforms({'img': scan})['img'] for scan in scans])
                    np.concatenate
                    # fold.append(transforms({
                    #     'img': info['scans'][:3],
                    #     'label': {
                    #         'WNT': 0,
                    #         'SHH': 1,
                    #         'G3': 2,
                    #         'G4': 3,
                    #     }[info['subgroup']],
                    # }))
                    bar.update()
                folds.append(fold)

    # for i, fold in enumerate(folds):
    #     folds[i] = map()

    return folds

if __name__ == '__main__':
    get_folds(None)
