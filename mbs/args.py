from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from umei.args import AugArgs, CVArgs, SegArgs, UMeIArgs

from mbs.utils.enums import Modality, SUBGROUPS, SegClass

@dataclass
class MBSegArgs(SegArgs, CVArgs, AugArgs, UMeIArgs):
    mc_seg: bool = field(default=True)
    z_strides: list[int] = field(default=None, metadata={'help': 'z-stride for each downsampling'})
    z_kernel_sizes: list[int] = field(default=None)
    input_modalities: list[Modality] = field(default=None, metadata={'choices': list(Modality)})
    pool_name: str = field(default='adaptiveavg', metadata={'choices': ['adaptiveavg', 'adaptivemax']})
    seg_classes: list[SegClass] = field(default=None, metadata={'choices': list(SegClass)})
    seg_weights: list[float] = field(default=None)
    test_size: int = field(default=None)
    pad_crop_size: list[int] = field(default=None)
    do_post: bool = field(default=False)
    train_cache_num: int = field(default=200)
    val_cache_num: int = field(default=50)
    train_batch_size: int = field(default=8)
    conv_norm: str = field(default='instance', metadata={'choices': ['instance', 'group', 'batch', 'syncbatch']})

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

@dataclass
class MBArgs(MBSegArgs):
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

    @property
    def cls_map(self):
        if self.cls_scheme == '4way':
            return {
                name: i
                for i, name in enumerate(SUBGROUPS)
            }
        if self.cls_scheme == '3way':
            return {
                'WNT': 0,
                'SHH': 1,
                'G3': 2,
                'G4': 2,
            }
        if self.cls_scheme == 'WS-G34':
            return {
                'WNT': 0,
                'SHH': 0,
                'G3': 1,
                'G4': 1,
            }
        if self.cls_scheme == 'WS':
            return {
                'WNT': 0,
                'SHH': 1,
                'G3': -1,
                'G4': -1,
            }
        if self.cls_scheme == 'G34':
            return {
                'WNT': -1,
                'SHH': -1,
                'G3': 0,
                'G4': 1,
            }

    @property
    def cls_names(self):
        if self.cls_scheme == '4way':
            return SUBGROUPS
        if self.cls_scheme == '3way':
            return ['WNT', 'SHH', 'G34']
        if self.cls_scheme == 'WS-G':
            return ['WS', 'G']
        if self.cls_scheme == 'WS':
            return ['WNT', 'SHH']
        if self.cls_scheme == 'G34':
            return ['G3', 'G4']

    @property
    def num_cls_classes(self):
        return len(self.cls_names)

    def __post_init__(self):
        super().__post_init__()
        if self.seg_seed is None:
            self.seg_seed = self.seed
        if self.cls_hidden_size is None:
            self.cls_hidden_size = self.feature_channels[-1] << 1

@dataclass
class MBSegPredArgs(MBSegArgs):
    p_seeds: list[int] = field(default=None)
    p_output_dir: Path = field(default=None)
    th: float = field(default=0.5)
    l: int = field(default=None)
    r: int = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        self.p_seeds = sorted(self.p_seeds)
        if self.p_output_dir is None:
            suffix = f'sw{self.sw_overlap}'
            if self.do_tta:
                suffix += '+tta'
            self.p_output_dir = self.output_dir / f'predict-{"+".join(map(str, self.p_seeds))}' / suffix
