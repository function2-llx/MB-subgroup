# from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field, fields
import multiprocessing
from pathlib import Path
from typing import Union

import transformers

from mbs.utils.dicom_utils import ScanProtocol

@dataclass
class TrainingArgs(transformers.TrainingArguments):
    """
    arguments for training & evaluation processes (model-agnostic)
    """
    exp_name: str = field(default=None)
    num_trials: int = field(default=5)
    output_dir: Path = field(default=None)
    dataloader_num_workers: int = field(default=multiprocessing.cpu_count())
    patience: int = field(default=5, metadata={'help': ''})
    num_folds: int = field(default=3, metadata={'help': 'number of folds for cross-validation'})
    amp: bool = field(default=True)

    img_dir: Path = field(default=None)
    seg_dir: Path = field(default=None)
    cohort_path: Path = field(default=None, metadata={'help': 'cohort file path, xlsx format'})

    conf_root: Path = field(default=Path('conf'))
    output_root: Path = field(default=Path('output'), metadata={'help': ''})

    @property
    def precision(self):
        return 16 if self.amp else 32

    @property
    def train_epochs(self) -> int:
        return int(self.num_train_epochs)

    def __post_init__(self):
        super().__post_init__()
        self.output_dir = Path(self.output_dir)
        self.output_root = Path(self.output_root)
        self.img_dir = Path(self.img_dir)
        self.seg_dir = Path(self.seg_dir)
        self.cohort_path = Path(self.cohort_path)
