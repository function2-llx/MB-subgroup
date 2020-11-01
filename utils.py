import os

import numpy as np
import png
from torchvision import transforms
from torchvision.datasets import ImageFolder


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


def split_ortn(path):
    # path will be orientation after the iteration. magic!
    for i in range(2):
        path = os.path.split(path)[i]
    return path


class MstFolder(ImageFolder):
    def __init__(self, root, transform, ortns):
        from torchvision.datasets.folder import is_image_file

        def is_valid_file(path):
            return is_image_file(path) and split_ortn(path) in ortns

        super().__init__(root, transform, is_valid_file=is_valid_file)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path = self.samples[index][0]
        # path will be orientation after the iteration. magic!
        for i in range(2):
            path = os.path.split(path)[i]
        return sample, path, target


def load_data(ortns, norm=True):
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
    image_datasets = {split: MstFolder(f'data/{split}', data_transforms[split], ortns=ortns) for split in ['train', 'val']}
    return image_datasets
