from typing import Sequence

from pytorch_lightning.trainer.states import RunningStage

from luolib.datamodule import SegDataModule
from luolib.utils import DataKey
from monai import transforms as monai_t

from mbs.conf import MBSegConf
from mbs.datamodule.base import MBDataModuleBase, load_merged_plan, load_split
from mbs.utils.enums import MBDataKey, SUBGROUPS

class MBSegDataModule(SegDataModule, MBDataModuleBase):
    conf: MBSegConf

    def load_data_transform(self, stage: RunningStage) -> list:
        match stage:
            case stage.PREDICTING:
                return [monai_t.LoadImageD(DataKey.IMG, ensure_channel_first=False, image_only=True)]
            case _:
                return [monai_t.LoadImageD([DataKey.IMG, DataKey.SEG], ensure_channel_first=False, image_only=True)]

    @property
    def split_cohort(self) -> dict[str, Sequence]:
        plan = load_merged_plan()
        split = load_split()
        data = {}
        for number, info in plan.iterrows():
            case_data_dir = self.conf.data_dir / number
            if (case_data_dir / f'{DataKey.SEG}.npy').exists():
                data.setdefault(split[number], []).append({
                    DataKey.CASE: number,
                    DataKey.CLS: SUBGROUPS.index(info['subgroup']),
                    **{
                        key: case_data_dir / f'{key}.npy'
                        for key in [DataKey.IMG, DataKey.SEG]
                    },
                    MBDataKey.SUBGROUP: info['subgroup'],
                })
        return data

    def intensity_normalize_transform(self, _stage):
        return []

    def spatial_normalize_transform(self, _stage):
        return []
