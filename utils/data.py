import json
import os
from typing import *

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import VisionDataset

targets = {
    'exists': ['no', 'yes'],
    'subgroup': ['WNT', 'SHH', 'G3', 'G4'],
    'subgroup2': ['WNT-SHH', 'G3-G4']
}


def get_ornt(ds) -> (str, int):
    ornt = np.array(ds.ImageOrientationPatient)
    p = sorted(np.argpartition(-abs(ornt), 2)[:2])
    if p == [0, 4]:
        return 'up', 2
    elif p == [1, 5]:
        return 'left', 0
    elif p == [0, 5]:
        return 'back', 1
    else:
        raise ValueError('cannot determine orientation')


def make_datasets(root, ortn, transform, data):
    samples = {
        target: []
        for target in targets.keys()
    }
    for patient, info in data.items():
        subgroup = info['subgroup_idx'] - 1
        subgroup2 = int(subgroup >= 2)
        for sample in info[ortn]:
            path = os.path.join(root, sample['path'])
            exists = sample['exists']
            samples['exists'].append((patient, path, int(exists)))
            if exists:
                samples['subgroup'].append((patient, path, subgroup))
                samples['subgroup2'].append((patient, path, subgroup2))
    return {
        target: ImageRecognitionDataset(root, transform, samples[target], len(target_names))
        for target, target_names in targets.items()
    }


class ImageRecognitionDataset(VisionDataset):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __init__(self, root, transform, samples: List[Tuple[str, str, int]], num_classes):
        super().__init__(root, transform=transform)
        self.samples = samples
        self.loader = torchvision.datasets.folder.default_loader
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        patient, path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        return patient, sample, target

    def collate_fn(self, batch):
        from torch.utils.data._utils.collate import default_collate
        return default_collate(batch)
        # inputs, targets, labels = list(zip(*batch))
        # return default_collate(inputs), default_collate(targets), labels

    def get_weight(self) -> torch.FloatTensor:
        weight = torch.zeros(self.num_classes)
        for _, _, target in self.samples:
            weight[target] += 1
        # weight.sqrt_()
        weight = weight.sum() / weight
        weight = weight / weight.sum()
        return weight
        # return torch.ones(4)

    # def get_weight(self) -> (torch.FloatTensor, torch.FloatTensor):
    #     weight_pos = torch.zeros(5)
    #     weight_neg = torch.zeros(5)
    #     for _, target in self.samples:
    #         exists = target['exists']
    #         subtype = target['subtype']
    #         if exists:
    #             weight_pos[subtype] += 1
    #             weight_pos[4] += 1
    #         else:
    #             weight_neg[subtype] += 1
    #             weight_neg[4] += 1
    #     tot = weight_pos + weight_neg
    #     weight_pos = tot / weight_pos
    #     weight_neg = tot / weight_neg
    #
    #     return torch.ones(5), torch.ones(5)


def load_data(data_dir, norm=True):
    data_transforms = {
        'train': [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ],
        'val': [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ],
    }
    if norm:
        for v in data_transforms.values():
            v.append(ImageRecognitionDataset.normalize)
    for k, v in data_transforms.items():
        data_transforms[k] = transforms.Compose(v)

    data_info = {
        split: json.load(open(os.path.join(data_dir, f'{split}.json')))
        for split in ['train', 'val']
    }
    image_datasets = {
        split: {
            ortn: make_datasets(data_dir, ortn, data_transforms[split], data_info[split])
            for ortn in ['back', 'up', 'left']
        } for split in ['train', 'val']
    }
    return image_datasets
