from typing import Optional, Tuple, Callable

import torch
from torch import nn
from torch.utils.data import Dataset

from resnet_3d.models.resnet import ResNet
from resnet_3d.utils import get_pretrain_config
from utils.data_3d import MultimodalDataset


class PairDataset(Dataset):
    def __init__(self, dataset: MultimodalDataset):
        self.dataset = dataset

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, float]:
        q, r = divmod(index, len(self.dataset))
        x, y = self.dataset[q], self.dataset[r]
        return x['img'], y['img'], float(x['label'] == y['label'])

    def __len__(self):
        return len(self.dataset) ** 2

class Siamese(ResNet):
    def setup_fc(self, n_output):
        from math import sqrt
        in_features = self.fc.in_features
        n_hidden = int(sqrt(in_features * n_output))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden, n_output),
        )

    # a trick make DataParallel have multiple methods, not sure if this will work fine
    def forward(self, impl: Callable[..., torch.Tensor], *args) -> torch.Tensor:
        return impl(*args)

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # def relation(self, x, y) -> torch.Tensor:
    #     r = self.fc.forward(torch.cat([x, y], dim=-1))
    #     return r.squeeze(-1)

    @classmethod
    def from_base(cls, base: ResNet):
        assert isinstance(base, ResNet)
        base.__class__ = Siamese
        assert isinstance(base, Siamese)
        return base


def generate_model(pretrained_name: str, n_output: Optional[int] = None) -> Siamese:
    from resnet_3d import model
    config = get_pretrain_config(pretrained_name)

    model = getattr(model, config['type']).generate_model(
        config['model_depth'],
        n_classes=config['n_pretrain_classes'] if n_output is None else n_output,
        n_input_channels=3,
    )
    return Siamese.from_base(model)


def load_pretrained_model(pretrained_name: str, n_output: Optional[int] = None) -> Siamese:
    from resnet_3d.model import load_pretrained_model, pretrained_root
    model = generate_model(pretrained_name, n_output)
    return load_pretrained_model(model, pretrained_root / f'{pretrained_name}.pth', 'resnet')
