from dataclasses import dataclass, field
from typing import List

@dataclass
class ClsUNetArgs:
    dim: int = field(default=3, metadata={'help': 'UNet dimension'})
    min_fmap: int = field(default=2, metadata={'help': 'Minimal dimension of feature map in the bottleneck'})
    depth: int = field(default=6, metadata={'help': 'The depth of the encoder'})
    num_cls_classes: int = field(default=None)
    num_seg_classes: int = field(default=None)
    filters: List[int] = field(default=None, metadata={'help': '[Optional] Set U-Net filters'})
    deep_supr_num: int = field(default=2, metadata={help: "Number of deep supervision heads"})
    res_block: bool = field(default=True, metadata={'help': "Enable residual blocks"})

    @property
    def num_input_channels(self) -> int:
        return 0

    @property
    def num_depth(self) -> int:
        """The depth of the encoder"""
        return len(self.filters) - 1
