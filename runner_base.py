import itertools
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import monai
import numpy as np
import torch
from monai.transforms import *

from utils.data import MultimodalDataset
from utils.report import Reporter
from utils.transforms import ToTensorDeviced

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
            ToTensorDeviced('img', self.args.device),
        ])
        val_set = MultimodalDataset(val_fold, val_transforms, len(self.args.target_names))
        return val_set

    def prepare_fold(self, val_id: int) -> Tuple[MultimodalDataset, MultimodalDataset]:
        train_folds = list(itertools.chain(*[fold for fold_id, fold in enumerate(self.folds) if fold_id != val_id]))
        train_transforms = {
            'no': [
                ToTensorDeviced('img', self.args.device),
            ],
            'weak': [
                RandFlipd('img', prob=0.5, spatial_axis=0),
                RandRotate90d('img', prob=0.5),
                ToTensorDeviced('img', self.args.device),
            ],
            'strong': [
                RandAffined(
                    'img',
                    mode="bilinear",
                    prob=0.8,
                    spatial_size=(self.args.sample_size, self.args.sample_size, self.args.sample_slices),
                    translate_range=(40, 40, 2),
                    rotate_range=(np.pi / 36, np.pi / 36, np.pi),
                    scale_range=(0.15, 0.15, 0.15),
                    padding_mode="border",
                ),
                ToTensorDeviced('img', self.args.device),
            ]
        }[self.args.aug]
        train_transforms = Compose(train_transforms)

        train_set = MultimodalDataset(train_folds, train_transforms, len(self.args.target_names))
        return train_set, self.prepare_val_fold(val_id)

    def run(self):
        for val_id in range(len(self.folds)):
            torch.cuda.empty_cache()
            self.run_fold(val_id)
            # update results after every fold
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
