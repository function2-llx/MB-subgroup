from dataclasses import dataclass, field
from pathlib import Path

# from umei.args import AugArgs, CVArgs, SegArgs, UMeIArgs
from umei.conf import SegExpConf, CrossValConf

from mbs.utils.enums import Modality, SUBGROUPS, SegClass

@dataclass(kw_only=True)
class MBSegConf(SegExpConf, CrossValConf):
    multi_label: bool = True
    # z_strides: list[int] = field(default=None, metadata={'help': 'z-stride for each downsampling'})
    # z_kernel_sizes: list[int] = field(default=None)
    input_modalities: list[Modality]
    pool_name: str = 'adaptiveavg'
    seg_classes: list[SegClass]
    seg_weights: list[float]
    test_size: int = field(default=None)
    pad_crop_size: list[int]
    do_post: bool = False
    train_cache_num: int = 200
    val_cache_num: int = 100
    train_batch_size: int = 8
    include_adults: bool = True

    def __post_init__(self):
        assert self.mc_seg
        if self.seg_weights is None:
            self.seg_weights = [1.] * self.num_seg_classes
        assert len(self.seg_weights) == self.num_seg_classes
        super().__post_init__()

    @property
    def num_input_channels(self) -> int:
        return len(self.input_modalities)

    @property
    def num_seg_classes(self) -> int:
        return len(self.seg_classes)

    @property
    def stem_stages(self) -> int:
        return self.num_stages - self.vit_stages

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
class MBClsConf(MBSegConf):
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

    @property
    def clinical_feature_size(self) -> int:
        return 4 * self.use_clinical
#
# @dataclass
# class MBSegPredArgs(MBSegArgs):
#     p_seeds: list[int] = field(default=None)
#     p_output_dir: Path = field(default=None)
#     th: float = field(default=0.5)
#     l: int = field(default=None)
#     r: int = field(default=None)
#
#     def __post_init__(self):
#         super().__post_init__()
#         self.p_seeds = sorted(self.p_seeds)
#         if self.p_output_dir is None:
#             suffix = f'sw{self.sw_overlap}'
#             if self.do_tta:
#                 suffix += '+tta'
#             self.p_output_dir = self.output_dir / f'predict-{"+".join(map(str, self.p_seeds))}' / suffix
