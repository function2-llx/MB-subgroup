import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from mashumaro import DataClassYAMLMixin

from utils.dicom_utils import ScanProtocol

conf_dir = Path(__file__).parent

@dataclass
class Conf(DataClassYAMLMixin):
    name: str
    n_folds: int
    folds_file: Path
    subgroups: List[str]
    protocols: List[ScanProtocol]
    sample_size: int
    sample_slices: int
    segs: List[str]
    model: str
    model_depth: int
    recons: bool
    lr: float
    aug: List[str]
    batch_size: int
    patience: int
    img_dir: Path
    seg_dir: Path
    do_train: bool = False
    force_retrain: bool = False
    epochs: int = 100
    lr_reduce_factor: float = 0.2
    resnet_shortcut: str = 'B'
    weight_decay: float = 0
    conv1_t_size: int = 7
    conv1_t_stride: int = 1
    resnet_widen_factor: float = 1
    no_max_pool: bool = False
    pretrain_name: Optional[str] = None
    device: str = None

    def __post_init__(self):
        self.folds_file = Path(self.folds_file)
        self.segs = list(map(str.upper, self.segs))
        for seg in self.segs:
            assert seg in ['AT', 'CT', 'WT']
        self.model_output_root = Path('output') / self.name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_dir = Path(self.img_dir)
        self.seg_dir = Path(self.seg_dir)

def get_conf() -> Conf:
    if len(sys.argv) == 2:
        with open(conf_dir / f'{sys.argv[1]}.yml') as f:
            return Conf.from_dict(yaml.safe_load(f))
    else:
        raise NotImplemented
