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
    pool_shape: list[int] = field(default_factory=lambda: [1, 1, 1])
    seg_classes: list[SegClass] = field(default=None, metadata={'choices': list(SegClass)})
    seg_weights: list[float] = field(default=None)
    test_size: int = field(default=None)
    pad_crop_size: list[int] = field(default=None)
    do_post: bool = field(default=False)
    train_cache_num: int = field(default=200)
    val_cache_num: int = field(default=50)
    train_batch_size: int = field(default=8)

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
    seg_pred_dir: Path = field(default=None)
    th: float = field(default=0.3)
    monitor: str = field(default='val/cls_loss')
    monitor_mode: str = field(default='min')
    per_device_eval_batch_size: int = field(default=4)
    cls_weights: list[float] = field(default=None)

    @property
    def num_cls_classes(self):
        return len(SUBGROUPS)

    def __post_init__(self):
        super().__post_init__()
        if self.seg_output_dir is None:
            self.seg_output_dir = self.output_dir
        if self.cls_weights is None:
            self.cls_weights = [1.] * self.num_cls_classes

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
