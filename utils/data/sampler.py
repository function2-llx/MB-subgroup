import random
from typing import Optional, Iterator

from torch.utils.data import Sampler
from torch.utils.data.sampler import T_co

from utils.data import MultimodalDataset


class BalancedSampler(Sampler):
    def __init__(self, dataset: MultimodalDataset, total: Optional[int] = None):
        super().__init__(dataset)
        self.indexes = {}
        for i, data in enumerate(dataset):
            self.indexes.setdefault(data['label'], []).append(i)
        self.labels = list(self.indexes.keys())

        if total is None:
            self.total = len(self.labels) * max([len(v) for v in self.indexes.values()])
        else:
            self.total = total

    def __iter__(self) -> Iterator[T_co]:
        pointers = {label: len(indexes) - 1 for label, indexes in self.indexes.items()}
        label_id = 0
        for _ in range(self.total):
            label = self.labels[label_id]
            pointers[label] += 1
            if pointers[label] == len(self.indexes[label]):
                pointers[label] = 0
                random.shuffle(self.indexes[label])
            yield self.indexes[label][pointers[label]]
            label_id = (label_id + 1) % len(self.labels)

    def __len__(self):
        return self.total
