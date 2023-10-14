from collections.abc import Hashable, Sequence
from typing import Any

import torch
from pytorch_lightning.trainer.states import RunningStage

from luolib.transforms.utils import FilterInstanceD
from luolib.utils import DataKey
from monai import transforms as mt
from monai.data.utils import collate_meta_tensor

from mbs.utils.enums import SegClass
from .seg_dm import MBSegDataModule
# from ..conf import MBM2FConf

class MBConvertUniversalSegmentationD(mt.Transform):
    def __init__(
        self,
        seg_key: Hashable = DataKey.SEG,
        cls_key: Hashable = DataKey.CLS,
        mask_key: Hashable = DataKey.SEG,
    ):
        self.seg_key = seg_key
        self.cls_key = cls_key
        self.mask_key = mask_key

    def __call__(self, data: dict[Hashable, Any]):
        d = dict(data)
        seg: torch.Tensor = d.pop(self.seg_key)
        class_label = torch.arange(1, seg.shape[0] + 1)
        d[self.cls_key] = class_label
        d[self.mask_key] = seg
        return d

class MBM2FDataModule(MBSegDataModule):
    # conf: MBM2FConf

    def train_collate_fn(self, batch: Sequence[dict]):
        elem = batch[0]
        ret = {}
        for key in elem:
            data_for_batch = [d[key] for d in batch]
            ret[key] = data_for_batch if key in [DataKey.CLS, DataKey.SEG] else collate_meta_tensor(data_for_batch)
        return ret

    def post_transform(self, stage: RunningStage):
        ret = super().post_transform(stage)
        match stage:
            case RunningStage.TRAINING:
                return ret + [
                    MBConvertUniversalSegmentationD(),
                    FilterInstanceD(class_key=DataKey.CLS, mask_key=DataKey.SEG),
                ]
            case _:
                return ret
