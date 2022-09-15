from dataclasses import dataclass, field

from mbs.utils.enums import Modality
from umei.args import AugArgs, CVArgs, SegArgs, UMeIArgs

@dataclass
class MBArgs(CVArgs, UMeIArgs):
    z_strides: list[int] = field(default=None, metadata={'help': 'z-stride for each downsampling'})
    input_modalities: list[Modality] = field(default=None, metadata={'choices': list(Modality)})
    test_size: int = field(default=None)

    @property
    def num_input_channels(self) -> int:
        return len(self.input_modalities)

    # @property
    # def num_stages(self) -> int:
    #     return len(self.z_strides) + 1

    @property
    def stem_stages(self) -> int:
        return self.num_stages - self.vit_stages

@dataclass
class MBSegArgs(SegArgs, MBArgs, AugArgs):  # be careful with the MRO
    mc_seg: bool = field(default=True)
    use_post: bool = field(default=False)

    @property
    def num_seg_classes(self) -> int:
        return 1
        # return len(SegClass)

    def __post_init__(self):
        assert self.mc_seg
        super().__post_init__()
