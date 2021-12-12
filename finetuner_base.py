import itertools
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, List

import monai
import numpy as np
import torch
from transformers import TrainingArguments, IntervalStrategy

from runner_base import RunnerBase
from utils.args import DataTrainingArgs, ModelArgs
from utils.data import MultimodalDataset
from utils.data.datasets.tiantan.args import MBArgs
from utils.report import Reporter

@dataclass
class FinetuneArgs(DataTrainingArgs, ModelArgs, MBArgs, TrainingArguments):
    folds_file: Path = field(default=None)
    num_pretrain_seg: int = field(default=None)
    patience: int = field(default=0)
    lr_reduce_factor: float = field(default=0.2)
    n_folds: int = None
    seg_inputs: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.save_strategy = IntervalStrategy.EPOCH
        super().__post_init__()
        self.folds_file = Path(self.folds_file)
        assert set(self.seg_inputs).issubset(set(self.segs))

class FinetunerBase(RunnerBase):
    args: FinetuneArgs

    def __init__(self, args: FinetuneArgs, folds):
        super().__init__(args)
        self.folds = folds
        # if args.rank == 0:
        logging.info(f"Training/evaluation parameters {args}")
        self.reporters: Dict[str, Reporter] = {
            test_name: Reporter(Path(args.output_dir) / test_name, args.subgroups, args.segs)
            for test_name in ['cross-val']
        }
        self.epoch_reporters: Dict[int, Dict[str, Reporter]] = {
            i: {
                test_name: Reporter(Path(args.output_dir) / 'epoch-reports' / f'ep{i}' / test_name, args.subgroups, args.segs)
                for test_name in ['cross-val']
            }
            for i in range(int(self.args.num_train_epochs) + 1)
        }

    def prepare_val_fold(self, val_id: int) -> MultimodalDataset:
        val_fold = self.folds[val_id]
        val_set = MultimodalDataset(
            val_fold,
            monai.transforms.Compose(self.get_inference_transforms(self.args)),
            len(self.args.subgroups),
        )
        return val_set

    def prepare_fold(self, val_id: int) -> Tuple[MultimodalDataset, MultimodalDataset]:
        train_folds = list(itertools.chain(*[fold for fold_id, fold in enumerate(self.folds) if fold_id != val_id]))

        train_set = MultimodalDataset(
            train_folds,
            monai.transforms.Compose(FinetunerBase.get_train_transforms(self.args)),
            len(self.args.subgroups),
        )
        return train_set, self.prepare_val_fold(val_id)

    def run(self):
        for val_id in range(len(self.folds)):
            torch.cuda.empty_cache()
            self.run_fold(val_id)
            # update results after every fold
            # if self.conf.rank == 0:
            # for reporter in self.reporters.values():
            #     reporter.report()

    @classmethod
    def get_train_transforms(cls, args: FinetuneArgs) -> List[monai.transforms.Transform]:
        all_keys = args.protocols + args.segs + ['fg_mask']
        ret: List[monai.transforms.Transform] = []

        if 'flip' in args.aug:
            ret.extend([
                monai.transforms.RandFlipD(all_keys, prob=0.5, spatial_axis=0),
                monai.transforms.RandFlipD(all_keys, prob=0.5, spatial_axis=1),
                monai.transforms.RandFlipD(all_keys, prob=0.5, spatial_axis=2),
                monai.transforms.RandRotate90D(all_keys, prob=0.5, max_k=1),
            ])
        if 'noise' in args.aug:
            ret.extend([
                monai.transforms.RandScaleIntensityD(args.protocols, factors=0.1, prob=1),
                monai.transforms.RandShiftIntensityD(args.protocols, offsets=0.1, prob=1),
            ])
        if 'blur' in args.aug:
            ret.append(monai.transforms.RandGaussianSmoothD(args.protocols, prob=0.5))

        return ret + cls.get_inference_transforms(args)

    @classmethod
    def get_inference_transforms(cls, args: FinetuneArgs) -> List[monai.transforms.Transform]:
        ret: List[monai.transforms.Transform] = []
        img_keys = args.protocols + args.seg_inputs
        if args.input_fg_mask:
            img_keys.append('fg_mask')
        ret.extend([
            monai.transforms.ConcatItemsD(img_keys, 'img'),
            monai.transforms.ConcatItemsD(args.segs, 'seg'),
            monai.transforms.CastToTypeD('img', np.float32),
            monai.transforms.CastToTypeD('seg', np.int),
            monai.transforms.ToTensorD(['img', 'seg']),
        ])

        return ret

    @abstractmethod
    def run_fold(self, val_id):
        pass
