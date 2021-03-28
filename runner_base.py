import itertools
import logging
from abc import ABC, abstractmethod
from typing import Tuple, List

import monai
import numpy as np
import torch
import torch.nn as nn
from monai.config import KeysCollection
from monai.transforms import *

from utils.data import MultimodalDataset
from utils.report import Reporter
from utils.to_tensor import ToTensorDeviced

class ConcatItemsd(MapTransform):
    """
    Concatenate specified items from data dictionary together on the first dim to construct a big array.
    Expect all the items are numpy array or PyTorch Tensor.

    """

    def __init__(self, keys: KeysCollection, name: str, dim: int = 0, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be concatenated together.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            name: the name corresponding to the key to store the concatenated data.
            dim: on which dimension to concatenate the items, default is 0.
            allow_missing_keys: don't raise exception if key is missing.

        Raises:
            ValueError: When insufficient keys are given (``len(self.keys) < 2``).

        """
        super().__init__(keys, allow_missing_keys)
        # if len(self.keys) < 2:
        #     raise ValueError("Concatenation requires at least 2 keys.")
        self.name = name
        self.dim = dim

    def __call__(self, data):
        """
        Raises:
            TypeError: When items in ``data`` differ in type.
            TypeError: When the item type is not in ``Union[numpy.ndarray, torch.Tensor]``.

        """
        d = dict(data)
        output = []
        data_type = None
        for key in self.key_iterator(d):
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")
            output.append(d[key])
        if data_type == np.ndarray:
            d[self.name] = np.concatenate(output, axis=self.dim)
        elif data_type == torch.Tensor:
            d[self.name] = torch.cat(output, dim=self.dim)
        else:
            raise TypeError(f"Unsupported data type: {data_type}, available options are (numpy.ndarray, torch.Tensor).")
        return d

class RunnerBase(ABC):
    def __init__(self, args, folds):
        self.args = args
        self.folds = folds
        self.setup_logging()
        if args.rank == 0:
            self.reporters = {
                test_name: Reporter(args.model_output_root / test_name, self.args.target_names)
                for test_name in ['cross-val']
            }

        if self.args.train:
            self.set_determinism()

    def setup_logging(self):
        args = self.args
        handlers = [logging.StreamHandler()]
        if args.train:
            args.model_output_root.mkdir(parents=True, exist_ok=True)
            mode = 'w' if args.rank == 0 else 'a'
            handlers.append(logging.FileHandler(args.model_output_root / 'train.log', mode))
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt=logging.Formatter.default_time_format,
            level=logging.INFO,
            handlers=handlers
        )

    def prepare_val_fold(self, val_id: int) -> MultimodalDataset:
        val_fold = self.folds[val_id]
        val_transforms = Compose([
            Resized(
                self.args.protocols,
                spatial_size=(self.args.sample_size, self.args.sample_size, self.args.sample_slices),
            ),
            ConcatItemsd(self.args.protocols, 'img'),
            SelectItemsd(['img', 'label']),
            ToTensorDeviced('img', self.args.device),
        ])
        val_set = MultimodalDataset(val_fold, val_transforms, len(self.args.target_names))
        return val_set

    def prepare_fold(self, val_id: int) -> Tuple[MultimodalDataset, MultimodalDataset]:
        train_folds = list(itertools.chain(*[fold for fold_id, fold in enumerate(self.folds) if fold_id != val_id]))
        aug: List[Transform] = []
        if self.args.aug == 'weak':
            aug = [
                RandFlipd(self.args.protocols, prob=0.5, spatial_axis=0),
                RandRotate90d(self.args.protocols, prob=0.5),
            ]
        elif self.args.aug == 'strong':
            raise NotImplementedError
        train_transforms = Compose(aug + [
            Resized(
                self.args.protocols,
                spatial_size=(self.args.sample_size, self.args.sample_size, self.args.sample_slices),
            ),
            ConcatItemsd(self.args.protocols, 'img'),
            SelectItemsd(['img', 'label']),
            ToTensorDeviced('img', self.args.device),
        ])
        train_set = MultimodalDataset(train_folds, train_transforms, len(self.args.target_names))
        return train_set, self.prepare_val_fold(val_id)

    def run(self):
        for val_id in range(len(self.folds)):
            self.run_fold(val_id)
            if self.args.rank == 0:
                for reporter in self.reporters.values():
                    reporter.report()

    def set_determinism(self):
        seed = self.args.seed
        monai.utils.set_determinism(seed)
        if self.args.rank == 0:
            logging.info(f'set random seed of {seed}\n')

    @abstractmethod
    def run_fold(self, val_id):
        pass
