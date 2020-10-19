import os

from torchvision import transforms
from torchvision.datasets import ImageFolder


class MstFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path = self.samples[index][0]
        # path will be orientation after the iteration. magic!
        for i in range(2):
            path = os.path.split(path)[i]
        return sample, path, target


def load_data():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {split: MstFolder(f'data/{split}', data_transforms[split]) for split in ['train', 'val']}
    return image_datasets
