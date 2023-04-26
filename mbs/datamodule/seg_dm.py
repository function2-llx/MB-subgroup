from typing import Sequence

from pytorch_lightning.trainer.states import RunningStage

from luolib.datamodule import SegDataModule
from luolib.utils import DataKey
from monai import transforms as monai_t

from mbs.conf import MBSegConf
from mbs.datamodule.base import MBDataModuleBase, load_merged_plan, load_split
from mbs.utils.enums import MBDataKey, SUBGROUPS

def _filter_seg(data: Sequence[dict]):
    return list(filter(lambda x: DataKey.SEG in x, data))

class MBSegDataModule(MBDataModuleBase, SegDataModule):
    conf: MBSegConf

    def load_data_transform(self, stage: RunningStage) -> list:
        match stage:
            case stage.PREDICTING:
                return [monai_t.LoadImageD(DataKey.IMG, ensure_channel_first=False, image_only=True)]
            case _:
                return [monai_t.LoadImageD([DataKey.IMG, DataKey.SEG], ensure_channel_first=False, image_only=True)]

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
