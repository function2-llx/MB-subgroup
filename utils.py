import os
import json

import numpy as np
import png
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import VisionDataset, DatasetFolder


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


def to_gray_scale(pixel_array: np.ndarray) -> np.ndarray:
    image_2d = pixel_array.astype(float)
    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    return image_2d_scaled


def write_gray_scale(pixel_array: np.ndarray, path: str):
    shape = pixel_array.shape
    with open(path, 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, pixel_array)


class MriDataset(VisionDataset):
    def __init__(self, root, ortn, transform, data):
        super().__init__(root, transform=transform)
        self.samples = []
        cnt = [0, 0]
        for patient, info in data.items():
            subtype = info['subtype_idx'] - 1
            for sample in info[ortn]:
                exists = sample['exists']
                cnt[exists] += 1
                self.samples.append([os.path.join(root, sample['path']), {
                    'exists': exists, 'subtype': subtype,
                }])
        print(ortn, cnt)
        self.loader = torchvision.datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        return sample, target

    def collate_fn(self, batch):
        from torch.utils.data._utils.collate import default_collate
        return default_collate(batch)
        # inputs, targets, labels = list(zip(*batch))
        # return default_collate(inputs), default_collate(targets), labels

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
            v.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    for k, v in data_transforms.items():
        data_transforms[k] = transforms.Compose(v)
    data_info = {
        split: json.load(open(os.path.join(data_dir, f'{split}.json')))
        for split in ['train', 'val']
    }
    image_datasets = {
        split: {
            ortn: MriDataset(data_dir, ortn, data_transforms[split], data_info[split])
            for ortn in ['back', 'up', 'left']
        } for split in ['train', 'val']
    }
    return image_datasets
