import itertools
import logging
from abc import ABC, abstractmethod
from typing import Tuple, List

import monai
import torch.nn as nn
from monai.transforms import *

from utils.data_3d import MultimodalDataset
from utils.dicom_utils import ScanProtocol
from utils.report import Reporter
from utils.to_tensor import ToTensorDeviced


class RunnerBase(ABC):
    def get_grouped_parameters(self, model: nn.Module):
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        no_decay = ['fc']
        grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-5,
             'lr': self.args.lr},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.args.lr * 10},
        ]
        return grouped_parameters

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
        modalities = list(ScanProtocol)
        val_fold = self.folds[val_id]
        val_transforms = Compose([
            Resized(
                modalities,
                spatial_size=(self.args.sample_size, self.args.sample_size, self.args.sample_slices),
            ),
            ConcatItemsd(modalities, 'img'),
            SelectItemsd(['img', 'label']),
            ToTensorDeviced('img', self.args.device),
        ])
        val_set = MultimodalDataset(val_fold, val_transforms, len(self.args.target_names))
        return val_set

    def prepare_fold(self, val_id: int) -> Tuple[MultimodalDataset, MultimodalDataset]:
        train_folds = list(itertools.chain(*[fold for fold_id, fold in enumerate(self.folds) if fold_id != val_id]))
        aug: List[Transform] = []
        modalities = list(ScanProtocol)
        if self.args.aug == 'weak':
            aug = [
                RandFlipd(modalities, prob=0.5, spatial_axis=0),
                RandRotate90d(modalities, prob=0.5),
            ]
        elif self.args.aug == 'strong':
            raise NotImplementedError
        train_transforms = Compose(aug + [
            Resized(
                modalities,
                spatial_size=(self.args.sample_size, self.args.sample_size, self.args.sample_slices),
            ),
            ConcatItemsd(modalities, 'img'),
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
