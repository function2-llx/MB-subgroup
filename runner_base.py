import itertools
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import monai
import numpy as np
import torch
import monai.transforms as monai_transforms

from utils.data import MultimodalDataset
from utils.report import Reporter
from utils.transforms import ToTensorDeviced, RandSpatialCropWithRatiod


class RunnerBase(ABC):
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        if self.args.train:
            self.set_determinism()

    def set_determinism(self):
        seed = self.args.seed
        monai.utils.set_determinism(seed)
        if self.is_world_master():
            logging.info(f'set random seed of {seed}\n')
        # not supported currently, throw RE for AdaptiveAvgPool
        # torch.use_deterministic_algorithms(True)

    def is_world_master(self) -> bool:
        return self.args.rank == 0

    def setup_logging(self):
        args = self.args
        handlers = [logging.StreamHandler()]
        if args.train:
            args.model_output_root.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(args.model_output_root / 'train.log', mode='a'))
            logging.basicConfig(
                format='%(asctime)s [%(levelname)s] %(message)s',
                datefmt=logging.Formatter.default_time_format,
                level=logging.INFO,
                handlers=handlers,
                force=True,
            )


class FinetunerBase(RunnerBase):
    def __init__(self, args, folds):
        super().__init__(args)
        self.folds = folds
        self.val_transforms = [
            monai_transforms.Resized(
                self.args.protocols,
                spatial_size=(self.args.sample_size, self.args.sample_size, -1),
            ),
            monai_transforms.ConcatItemsd(self.args.protocols, 'img'),
            monai_transforms.SelectItemsd(keys=['img', 'label']),
            monai_transforms.NormalizeIntensityd('img', channel_wise=True, nonzero=True),
            ToTensorDeviced('img', self.args.device),
        ]
        if args.rank == 0:
            self.reporters = {
                test_name: Reporter(args.model_output_root / test_name, self.args.target_names)
                for test_name in ['cross-val']
            }

    def prepare_val_fold(self, val_id: int) -> MultimodalDataset:
        val_fold = self.folds[val_id]
        val_set = MultimodalDataset(
            val_fold,
            monai_transforms.Compose(self.val_transforms),
            len(self.args.target_names),
        )
        return val_set

    def prepare_fold(self, val_id: int) -> Tuple[MultimodalDataset, MultimodalDataset]:
        train_folds = list(itertools.chain(*[fold for fold_id, fold in enumerate(self.folds) if fold_id != val_id]))
        train_transforms = {
            'no': self.val_transforms,
            'weak': [
                monai_transforms.Resized(
                    self.args.protocols,
                    spatial_size=(self.args.sample_size, self.args.sample_size, -1),
                ),
                monai_transforms.ConcatItemsd(self.args.protocols, 'img'),
                monai_transforms.SelectItemsd(keys=['img', 'label']),
                monai_transforms.RandFlipd(keys='img', prob=0.5, spatial_axis=0),
                monai_transforms.RandRotate90d(keys='img', prob=0.5),
                monai_transforms.NormalizeIntensityd(keys='img', channel_wise=True, nonzero=True),
                ToTensorDeviced('img', self.args.device),
            ],
            'strong': [
                # randomly crop selected slices
                RandSpatialCropWithRatiod(
                    keys=self.args.protocols,
                    roi_ratio=(self.args.crop_ratio, self.args.crop_ratio, 1),
                    random_center=True,
                    random_size=True,
                ),
                monai_transforms.Resized(
                    keys=self.args.protocols,
                    spatial_size=(self.args.sample_size, self.args.sample_size, -1),
                ),
                monai_transforms.ConcatItemsd(keys=self.args.protocols, name='img'),
                monai_transforms.SelectItemsd(keys=['img', 'label']),
                monai_transforms.RandFlipd(keys='img', prob=0.5, spatial_axis=0),
                monai_transforms.RandRotate90d(keys='img', prob=0.5),
                monai_transforms.NormalizeIntensityd(keys='img', channel_wise=True, nonzero=True),
                monai_transforms.RandScaleIntensityd(keys='img', factors=0.1, prob=0.5),
                monai_transforms.RandShiftIntensityd(keys='img', offsets=0.1, prob=0.5),
                ToTensorDeviced(keys='img', device=self.args.device),
            ]
        }[self.args.aug]

        train_set = MultimodalDataset(
            train_folds,
            monai_transforms.Compose(train_transforms),
            len(self.args.target_names),
        )
        return train_set, self.prepare_val_fold(val_id)

    def run(self):
        for val_id in range(len(self.folds)):
            torch.cuda.empty_cache()
            self.run_fold(val_id)
            # update results after every fold
            if self.args.rank == 0:
                for reporter in self.reporters.values():
                    reporter.report()

    @abstractmethod
    def run_fold(self, val_id):
        pass
