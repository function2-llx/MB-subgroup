# from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import transformers

from utils.dicom_utils import ScanProtocol

@dataclass
class TrainingArgs(transformers.TrainingArguments):
    """
    arguments for training & evaluation processes (model-agnostic)
    """
    patience: int = field(default=5, metadata={'help': ''})
    num_folds: int = field(default=3, metadata={'help': 'number of folds for cross-validation'})
    amp: bool = field(default=True)

    img_dir: Path = field(default=None)
    seg_dir: Path = field(default=None)
    cohort_path: Path = field(default=None, metadata={'help': 'cohort file path, xlsx format'})

    conf_root: Path = field(default=Path('conf'))
    output_root: Path = field(default=Path('output'), metadata={'help': ''})

    @property
    def train_epochs(self) -> int:
        return int(self.num_train_epochs)

    def __post_init__(self):
        super().__post_init__()
        self.img_dir = Path(self.img_dir)
        self.seg_dir = Path(self.seg_dir)
        self.cohort_path = Path(self.cohort_path)

@dataclass
class DataTrainingArgs:
    sample_size: int = field(default=None)
    sample_slices: int = field(default=None)
    aug: list[str] = field(default=None)
    subjects: int = field(default=None)
    modalities: list[Union[str, ScanProtocol]] = field(default_factory=lambda: [protocol.name for protocol in list(ScanProtocol)])
    input_fg_mask: bool = field(default=True)
    use_focal: bool = field(default=False)
    do_ensemble: bool = field(default=False)

    @property
    def in_channels(self) -> int:
        return len(self.modalities) + self.input_fg_mask

@dataclass
class ModelArgs:
    model: str = field(default=None)
    model_name_or_path: str = field(default=None)
    resnet_shortcut: str = field(default='B')
    conv1_t_size: int = 7
    model_depth: int = 34
    conv1_t_stride: int = 1
    resnet_widen_factor: float = 1
    no_max_pool: bool = False
    cls_factor: float = None
    seg_factor: float = None
    vae_factor: float = None
