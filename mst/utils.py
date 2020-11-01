import os

from torchvision import transforms
from torchvision.datasets import ImageFolder


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
