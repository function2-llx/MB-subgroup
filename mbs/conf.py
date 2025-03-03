import json
from dataclasses import dataclass
from pathlib import Path

import torch.cuda
# from omegaconf import II, OmegaConf

# from luolib.conf import ClsExpConf, CrossValConf, SegExpConf, Mask2FormerConf, SegCommonConf, OptimizerConf, parse_exp_conf
from luolib.types import tuple3_t

from mbs.utils.enums import SUBGROUPS, SegClass, PROCESSED_DIR

# @dataclass(kw_only=True)
# class MBConfBase(CrossValConf):
#     num_input_channels: int = 3
#     include_adults: bool = True
#     data_dir: Path = PROCESSED_DIR / 'cr-p10/normalized'
#
# @dataclass(kw_only=True)
# class MBSegConfBase(MBConfBase):
#     seg_classes: list[SegClass]
#
# @dataclass(kw_only=True)
# class MBSegConf(MBSegConfBase, SegExpConf):
#     conf_root: Path = Path('conf/tasks/seg')
#     output_root: Path = Path('output/seg')
#     num_seg_classes: int = 3
#     multi_label: bool = True
#     seg_weights: list[float] | None = None
#     # test_size: int = field(default=None)
#     # pad_crop_size: list[int]
#     train_cache_num: int = 200
#     val_cache_num: int = 100

# @dataclass(kw_only=True)
# class MBSegPredConf(SegCommonConf):
#     exp_conf_cls: str
#     exp_conf_path: Path
#     exp_conf: MBSegConfBase
#     datamodule_cls: str
#     inferer_cls: str
#
#     p_seeds: list[int]
#     p_output_dir: Path | None = None
#     th: float = 0.5
#     overwrite: bool = False
#     all_folds: bool = False
#     l: int | None = None
#     r: int | None = None
#
#     def default_pred_output_dir(self):
#         if self.p_output_dir is None:
#             suffix = f'sw{self.sw_overlap}'
#             if self.exp_conf.do_tta:
#                 suffix += '+tta'
#             self.p_output_dir = self.exp_conf.output_dir / f'predict-{"+".join(map(str, self.p_seeds))}' / suffix
#
#     def get_sub(self, post: bool):
#         suffix = f'th{self.th}'
#         if post:
#             suffix += '-post'
#         return suffix
#
#     def get_case_save_dir(self, case: str):
#         return self.p_output_dir / 'pred' / case
#
#     def get_save_path(self, case: str, seg_class: SegClass, post: bool, suffix: str = '.pt'):
#         return MBSegPredConf.get_case_save_dir(self, case) / MBSegPredConf.get_sub(self, post) / f'{seg_class}{suffix}'
#
#     def load_exp_conf(self):
#         assert self.exp_conf_cls in [MBM2FConf.__name__, MBSegConf.__name__]
#         exp_conf_cls = globals()[self.exp_conf_cls]
#         self.exp_conf = OmegaConf.merge(    # type: ignore
#             parse_exp_conf(exp_conf_cls, self.exp_conf_path),
#             # make omegaconf happy, or it will complain that the class of `conf.exp_conf` is not subclass of `exp_conf_cls`
#             OmegaConf.to_container(self.exp_conf),
#         )

def get_cls_map(cls_scheme: str) -> dict[str, int]:
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

def get_cls_map_vec(scheme: str, device='cuda') -> torch.Tensor:
    cls_map = get_cls_map(scheme)
    return torch.tensor([cls_map[subgroup] for subgroup in SUBGROUPS], device=device)

def get_cls_names(cls_scheme: str) -> list[str]:
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

# @dataclass(kw_only=True)
# class MBClsConf(MBConfBase, ClsExpConf):
#     conf_root: Path = Path('conf/tasks/cls')
#     output_root: Path = Path('output/cls')
#     monitor: str = 'val/loss'
#     monitor_mode: str = 'min'
#     # include seed
#     pretrain_cv_dir: Path | None = None
#     pretrain_ckpt_path: Path | None = None
#     seg_pred_dir: Path
#     th: float = 0.5
#     use_post: bool
#     eval_batch_size: int = 16 * torch.cuda.device_count()
#     val_test: bool = True
#     # choices: 4way, 3way, WS-G34, WS, G34
#     cls_scheme: str = '4way'
#     pred_keys: list[str]
#     use_clinical: bool = False
#
#     pool_types: list[str]
#     pooling_layer: str
#     pooling_level_stride: list[int]
#     pooling_th: int
#
#     def get_pred_path(self, case: str, seg_class: SegClass, suffix: str = '.pt'):
#         return self.seg_pred_dir / 'pred' / case / MBSegPredConf.get_sub(self, self.use_post) / f'{seg_class}{suffix}'
#
#     def load_center(self) -> dict[str, tuple3_t[int]]:
#         center_file_path = self.seg_pred_dir / 'center' / f'{MBSegPredConf.get_sub(self, self.use_post)}.json'
#         center = json.loads(center_file_path.read_bytes())
#         for k, v in center.items():
#             center[k] = tuple(v)
#         return center

# @dataclass(kw_only=True)
# class MBM2FConf(MBSegConfBase, Mask2FormerConf):
#     conf_root: Path = Path('conf/tasks/m2f')
#     output_root: Path = Path('output/m2f')
#     max_epochs: int | None = None
#     train_cache_num: int = 200
#     val_cache_num: int = 100
#     num_fg_classes: int = 2
#     monitor: str = 'val/dice/avg'
#     monitor_mode: str = 'max'
#     multi_label: bool = True
#     num_seg_classes: int = II('.num_fg_classes')
#
# @dataclass(kw_only=True)
# class MBM2FClsConf(MBClsConf, Mask2FormerConf):
#     cls_head_optim: OptimizerConf
