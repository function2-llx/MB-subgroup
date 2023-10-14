from typing import Sequence

from pytorch_lightning.trainer.states import RunningStage

from luolib.datamodule import SegDataModule
from luolib.utils import DataKey
from monai import transforms as mt

# from mbs.conf import MBSegConf
from mbs.datamodule import MBDataModuleBase
from mbs.utils.enums import SegClass

def _filter_seg(data: Sequence[dict]):
    return list(filter(lambda x: DataKey.SEG in x, data))

class MBSegDataModule(MBDataModuleBase, SegDataModule):
    # conf: MBSegConf

    def load_data_transform(self, stage: RunningStage):
        match stage:
            case stage.PREDICTING:
                return [mt.LoadImageD(DataKey.IMG, ensure_channel_first=False, image_only=True)]
            case _:
                return [
                    mt.LoadImageD([DataKey.IMG, DataKey.SEG], ensure_channel_first=False, image_only=True),
                    mt.LambdaD(DataKey.SEG, lambda seg: seg[[
                        list(SegClass).index(seg_class)
                        for seg_class in self.conf.seg_classes
                    ]]),
                ]

    def intensity_normalize_transform(self, _stage):
        return []

    def spatial_normalize_transform(self, _stage):
        return []

    def train_data(self):
        return _filter_seg(super().train_data())

    def val_data(self):
        return _filter_seg(super().val_data())

    def test_data(self):
        return _filter_seg(super().test_data())
