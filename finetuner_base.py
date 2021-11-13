import itertools
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Tuple, Dict

import torch
from monai import transforms as monai_transforms

from runner_base import RunnerBase
from utils.args import FinetuneArgs
from utils.data import MultimodalDataset
from utils.report import Reporter

class FinetunerBase(RunnerBase):
    args: FinetuneArgs

    def __init__(self, args: FinetuneArgs, folds):
        super().__init__(args)
        self.folds = folds
        # if args.rank == 0:
        logging.info(f"Training/evaluation parameters {args}")
        self.reporters: Dict[str, Reporter] = {
            test_name: Reporter(Path(args.output_dir) / test_name, args.subgroups)
            for test_name in ['cross-val']
        }

    def prepare_val_fold(self, val_id: int) -> MultimodalDataset:
        val_fold = self.folds[val_id]
        val_set = MultimodalDataset(
            val_fold,
            monai_transforms.Compose(RunnerBase.get_inference_transforms(self.args)),
            len(self.args.subgroups),
        )
        return val_set

    def prepare_fold(self, val_id: int) -> Tuple[MultimodalDataset, MultimodalDataset]:
        train_folds = list(itertools.chain(*[fold for fold_id, fold in enumerate(self.folds) if fold_id != val_id]))

        train_set = MultimodalDataset(
            train_folds,
            monai_transforms.Compose(RunnerBase.get_train_transforms(self.args)),
            len(self.args.subgroups),
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
