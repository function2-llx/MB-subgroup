from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from umei.args import AugArgs, CVArgs, SegArgs, UMeIArgs

from mbs.utils.enums import Modality, SUBGROUPS, SegClass

@dataclass
class MBSegArgs(SegArgs, CVArgs, AugArgs, UMeIArgs):
    mc_seg: bool = field(default=True)
    z_strides: list[int] = field(default=None, metadata={'help': 'z-stride for each downsampling'})
    input_modalities: list[Modality] = field(default=None, metadata={'choices': list(Modality)})
    seg_classes: list[SegClass] = field(default=None, metadata={'choices': list(SegClass)})
    test_size: int = field(default=None)
    pad_crop_size: list[int] = field(default=None)
    do_post: bool = field(default=False)
    train_cache_num: int = field(default=200)
    val_cache_num: int = field(default=50)
    train_batch_size: int = field(default=8)

    def __post_init__(self):
        assert self.mc_seg
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
    seg_pred_dir: Path = field(default=None)
    th: float = field(default=0.3)
    do_post: bool = field(default=True)
    monitor: str = field(default='val/auc/avg')
    monitor_mode: str = field(default='max')
    learning_rate: float = field(default=1e-5)
    warmup_epochs: int = field(default=0)
    seg_loss_factor: float = field(default=0.2)
    num_train_epochs: float = field(default=50)

    @property
    def num_cls_classes(self):
        return len(SUBGROUPS)

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
