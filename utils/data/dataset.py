from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

import torch

from monai.data import CacheDataset

class MultimodalDataset(CacheDataset):
    def __init__(
        self,
        data: list[dict],
        transform: Sequence[Callable] | Callable,
        num_classes: Optional[int] = None,
        **kwargs,
    ):
        if issubclass(MultimodalDataset, CacheDataset):
            super().__init__(data, transform, **kwargs)
        else:
            super().__init__(data, transform)
        self.keys = data[0].keys()
        for x in data:
            assert self.keys == x.keys()

        self.num_classes = num_classes

    def get_weight(self, strategy) -> torch.Tensor:
        assert 'label' in self.keys and self.num_classes is not None
        labels = [data['label'] for data in self.data]
        if strategy == 'equal':
            return torch.ones(self.num_classes)
        elif strategy == 'invsqrt':
            weight = torch.zeros(self.num_classes)
            for label in labels:
                weight[label] += 1
            weight.sqrt_()
            weight = weight.sum() / weight
            weight = weight / weight.sum()
            return weight
        elif strategy == 'inv':
            weight = torch.zeros(self.num_classes)
            for label in labels:
                weight[label] += 1
            # weight.sqrt_()
            weight = weight.sum() / weight
            weight = weight / weight.sum()
            return weight
        else:
            raise ValueError(f'unsupported weight strategy of {strategy}')
