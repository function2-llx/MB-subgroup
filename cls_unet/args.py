from dataclasses import dataclass, field
from typing import Literal

from transformers import TrainingArguments

@dataclass
class ClsUNetArgs(TrainingArguments):
    dim: int = field(default=3, metadata={'help': 'UNet spatial dimensions'})
    min_fmap: int = field(default=2, metadata={'help': 'Minimal dimension of feature map in the bottleneck'})
    pool_type: Literal['max', 'avg'] = field(default='max')
    pool_fmap: int = field(default=2, metadata={'help': 'feature map size after pooling for classification'})

    sample_size: int = field(default=302, metadata={'help': 'image size of single slice'})
    sample_slices: int = field(default=16, metadata={'help': 'number of slices'})
    spacing: tuple[float, ...] = field(default=None)

    modalities: list[str] = field(default=None)
    cls_labels: list[str] = field(default=None)
    seg_labels: list[str] = field(default=None)

    filters: list[int] = field(default=None, metadata={'help': '[Optional] Set U-Net filters'})
    deep_supr_num: int = field(default=2, metadata={help: "Number of deep supervision heads"})
    res_block: bool = field(default=True, metadata={'help': "Enable residual blocks"})


    def __post_init__(self):
        self.spacing = tuple(self.spacing)

    @property
    def num_input_channels(self) -> int:
        return len(self.modalities)

    @property
    def num_cls_classes(self) -> int:
        return len(self.cls_labels)

    @property
    def num_seg_classes(self) -> int:
        return len(self.seg_labels)

    @property
    def num_depth(self) -> int:
        """The depth of the encoder"""
        return len(self.filters) - 1

    @property
    def patch_size(self) -> tuple[int, ...]:
        # implement for 3d
        return self.sample_size, self.sample_size, self.sample_slices
