from dataclasses import dataclass, field
from pathlib import Path

from luolib.conf import ClsExpConf, CrossValConf, SegExpConf

from mbs.utils.enums import SUBGROUPS, SegClass, PROCESSED_DIR

@dataclass(kw_only=True)
class MBConfBase(CrossValConf):
    num_input_channels: int = 3
    include_adults: bool = True
    data_dir: Path = PROCESSED_DIR / 'cr-p10/normalized'

@dataclass(kw_only=True)
class MBSegConf(MBConfBase, SegExpConf):
    conf_root: Path = Path('conf/tasks/seg')
    output_root: Path = Path('output/seg')
    num_seg_classes: int = 3
    multi_label: bool = True
    seg_weights: list[float] | None = None
    # test_size: int = field(default=None)
    # pad_crop_size: list[int]
    train_cache_num: int = 200
    val_cache_num: int = 100

@dataclass(kw_only=True)
class MBSegPredConf(MBSegConf):
    p_seeds: list[int]
    p_output_dir: Path | None = None
    th: float = 0.5
    overwrite: bool = False
    l: int | None = None
    r: int | None = None

def cls_map(cls_scheme: str) -> dict[str, int]:
    match cls_scheme:
        case '4way':
            return {
                name: i
                for i, name in enumerate(SUBGROUPS)
            }
        case '3way':
            return {
                'WNT': 0,
                'SHH': 1,
                'G3': 2,
                'G4': 2,
            }
        case 'WS-G34':
            return {
                'WNT': 0,
                'SHH': 0,
                'G3': 1,
                'G4': 1,
            }
        case 'WS':
            return {
                'WNT': 0,
                'SHH': 1,
                'G3': -1,
                'G4': -1,
            }
        case 'G34':
            return {
                'WNT': -1,
                'SHH': -1,
                'G3': 0,
                'G4': 1,
            }
        case _:
            raise ValueError

def cls_names(cls_scheme: str) -> list[str]:
    match cls_scheme:
        case '4way':
            return SUBGROUPS
        case '3way':
            return ['WNT', 'SHH', 'G34']
        case 'WS-G34':
            return ['WS', 'G34']
        case 'WS':
            return ['WNT', 'SHH']
        case 'G34':
            return ['G3', 'G4']
        case _:
            raise ValueError

@dataclass
class MBClsConf(MBConfBase, ClsExpConf):
    seg_output_dir: Path = field(default=None)
    seg_seed: int = field(default=None)
    seg_pred_dir: Path = field(default=None)
    th: float = field(default=0.3)
    monitor: str = field(default='val/cls_loss')
    monitor_mode: str = field(default='min')
    per_device_eval_batch_size: int = field(default=4)
    cls_weights: list[float] = field(default=None)
    finetune_lr: float = field(default=1e-4)
    cls_conv: bool = field(default=True)
    cls_hidden_size: int = field(default=None)
    addi_conv: bool = field(default=False)
    cls_scheme: str = field(default='4way', metadata={'choices': ['4way', '3way', 'WS-G34', 'WS', 'G34']})
    seg_inputs: list[str] = field(default_factory=list)
    crop_ref: SegClass = field(default=SegClass.ST)
    use_clinical: bool = field(default=False)
