import itertools
from abc import abstractmethod
from pathlib import Path
from typing import Tuple

import torch
from monai import transforms as monai_transforms

from runner_base import RunnerBase
from utils.conf import Conf
from utils.data import MultimodalDataset
from utils.report import Reporter

class FinetunerBase(RunnerBase):
    def __init__(self, conf: Conf, folds):
        super().__init__(conf)
        self.folds = folds
        # if args.rank == 0:
        self.reporters = {
            test_name: Reporter(Path('output') / conf.name / test_name, conf.subgroups)
            for test_name in ['cross-val']
        }

    def prepare_val_fold(self, val_id: int) -> MultimodalDataset:
        val_fold = self.folds[val_id]
        val_set = MultimodalDataset(
            val_fold,
            monai_transforms.Compose(RunnerBase.get_inference_transforms(self.conf)),
            len(self.conf.subgroups),
        )
        return val_set

    def prepare_fold(self, val_id: int) -> Tuple[MultimodalDataset, MultimodalDataset]:
        train_folds = list(itertools.chain(*[fold for fold_id, fold in enumerate(self.folds) if fold_id != val_id]))

        train_set = MultimodalDataset(
            train_folds,
            monai_transforms.Compose(RunnerBase.get_train_transforms(self.conf)),
            len(self.conf.subgroups),
        )
        return train_set, self.prepare_val_fold(val_id)

    def run(self):
        for val_id in range(len(self.folds)):
            torch.cuda.empty_cache()
            self.run_fold(val_id)
            # update results after every fold
            # if self.conf.rank == 0:
            for reporter in self.reporters.values():
                reporter.report()

    @abstractmethod
    def run_fold(self, val_id):
        pass
