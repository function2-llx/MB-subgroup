from __future__ import annotations

from dataclasses import dataclass, field

import transformers

from utils.dicom_utils import ScanProtocol

@dataclass
class TrainingArgs(transformers.TrainingArguments):
    patience: int = field(default=5)
    num_folds: int = field(default=3, metadata={'help': 'number of folds for cross-validation'})
    sync_batchnorm: bool = field(default=False, metadata={'help': 'Enable synchronized batchnorm'})

@dataclass
class DataTrainingArgs:
    sample_size: int = field(default=None)
    sample_slices: int = field(default=None)
    aug: list[str] = field(default=None)
    subjects: int = field(default=None)
    protocols: list[str | ScanProtocol] = field(default_factory=lambda: [protocol.name for protocol in list(ScanProtocol)])
    input_fg_mask: bool = field(default=True)
    use_focal: bool = field(default=False)
    do_ensemble: bool = field(default=False)

    @property
    def in_channels(self) -> int:
        ret = len(self.protocols)
        if self.input_fg_mask:
            ret += 1
        return ret

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
