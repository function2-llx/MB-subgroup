from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

import monai
from monai.utils import InterpolateMode

from utils.transforms import CreateForegroundMaskD

@dataclass
class ClsUNetArgs:
    """
    arguments related to the model
    """
    dim: int = field(default=3, metadata={'help': 'UNet spatial dimensions'})
    min_fmap: int = field(default=2, metadata={'help': 'Minimal dimension of feature map in the bottleneck'})
    pool_type: Literal['max', 'avg'] = field(default='max')
    pool_fmap: int = field(default=2, metadata={'help': 'feature map size after pooling for classification'})

    sample_size: int = field(default=302, metadata={'help': 'image size of single slice'})
    sample_slices: int = field(default=16, metadata={'help': 'number of slices'})
    spacing: tuple[float, ...] = field(default=None)

    modalities: list[str] = field(default=None)
    input_fg_mask: bool = field(default=False, metadata={'help': 'if using one-hot encoding for foreground voxels (voxel value > 0) as additional input'})
    img_key: str = field(default='img')
    seg_key: str = field(default='seg')
    cls_key: str = field(default='cls')
    fg_mask_key: str = field(default='fg_mask', metadata={'help': 'key name of one-hot encoding for foreground voxels (voxel value > 0)'})
    cls_labels: list[str] = field(default=None)
    seg_labels: list[str] = field(default=None)

    filters: list[int] = field(default=None, metadata={'help': '[Optional] Set U-Net filters'})
    deep_supr_num: int = field(default=2, metadata={help: "Number of deep supervision heads"})
    res_block: bool = field(default=True, metadata={'help': "Enable residual blocks"})

    data_dir: Path = field(default=None)

    def __post_init__(self):
        self.spacing = tuple(self.spacing)

    @property
    def num_input_channels(self) -> int:
        return len(self.modalities) + self.input_fg_mask

    @property
    def all_data_keys(self) -> list[str]:
        ret = self.modalities + self.seg_labels
        if self.fg_mask_key is not None:
            ret.append(self.fg_mask_key)
        return ret

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

    @property
    def preprocess_transform(self) -> Callable:
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(self.modalities + self.seg_labels),
            monai.transforms.AddChannelD(self.modalities + self.seg_labels),
            monai.transforms.OrientationD(self.modalities + self.seg_labels, 'RAS'),
            monai.transforms.SpatialCropD(
                self.modalities + self.seg_labels,
                roi_slices=[slice(None), slice(None), slice(0, self.sample_slices)],
            ),
            monai.transforms.ResizeD(self.modalities, spatial_size=(self.sample_size, self.sample_size, -1)),
            monai.transforms.ResizeD(
                self.seg_labels,
                spatial_size=(self.sample_size, self.sample_size, -1),
                mode=InterpolateMode.NEAREST,
            ),
            monai.transforms.ThresholdIntensityD(self.modalities, threshold=0),
            CreateForegroundMaskD(self.modalities, mask_key=self.fg_mask_key),
            monai.transforms.NormalizeIntensityD(self.modalities, nonzero=self.input_fg_mask),
            monai.transforms.ThresholdIntensityD(self.seg_labels, threshold=1, above=False, cval=1),
        ])

    @property
    def aug_transform(self) -> Callable:
        keys = self.all_data_keys
        return monai.transforms.Compose([
            monai.transforms.RandFlipD(keys, prob=0.5, spatial_axis=0),
            monai.transforms.RandFlipD(keys, prob=0.5, spatial_axis=1),
            monai.transforms.RandFlipD(keys, prob=0.5, spatial_axis=2),
            monai.transforms.RandRotate90D(keys, prob=0.5, max_k=1, spatial_axes=(0, 1)),
        ])

    @property
    def input_transform(self) -> Callable:
        img_keys = copy(self.modalities)
        if self.input_fg_mask:
            img_keys.append(self.fg_mask_key)
        return monai.transforms.Compose([
            monai.transforms.ConcatItemsD(img_keys, self.img_key),
            monai.transforms.ConcatItemsD(self.seg_labels, self.seg_key),
            monai.transforms.CastToTypeD(self.img_key, np.float32),
            monai.transforms.CastToTypeD(self.seg_key, np.int),
            # monai.transforms.ToTensorD([self.img_key, self.seg_key, self.cls_key]),
        ])

    @property
    def train_transform(self) -> Callable:
        return monai.transforms.Compose([
            self.preprocess_transform,
            self.aug_transform,
            self.input_transform,
        ])

    @property
    def eval_transform(self):
        return monai.transforms.Compose([
            self.preprocess_transform,
            self.input_transform,
        ])
