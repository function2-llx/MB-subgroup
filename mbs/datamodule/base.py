from collections.abc import Hashable
from functools import cached_property
import itertools
import math
from typing import Sequence

import pandas as pd

from luolib.datamodule import CrossValDataModule
from luolib.datamodule.base import DataSeq
from luolib.utils import DataSplit, DataKey

from mbs.conf import MBConfBase
from mbs.utils.enums import CLINICAL_DIR, MBDataKey, PROCESSED_DIR, SUBGROUPS


def load_clinical():
    clinical = pd.read_excel(CLINICAL_DIR / 'clinical-com.xlsx', dtype='string').set_index('住院号')
    return clinical

class MBDataModuleBase(CrossValDataModule):
    conf: MBConfBase

    @cached_property
    def split_cohort(self) -> dict[Hashable, DataSeq]:
        plan = load_merged_plan()
        split = load_split()
        split_cohort = {}
        for number, info in plan.iterrows():
            case_data_dir = self.conf.data_dir / number
            split_cohort.setdefault(split[number], []).append({
                DataKey.CASE: number,
                **{
                    key: path
                    for key in [DataKey.IMG, DataKey.SEG] if (path := case_data_dir / f'{key}.npy').exists()
                },
                MBDataKey.SUBGROUP: info['subgroup'],
            })
        return split_cohort

    @cached_property
    def partitions(self):
        ret = [
            self.split_cohort[fold_id]
            for fold_id in range(self.conf.num_folds)
        ]
        # trick: select training data for fold-i is by deleting the i-th item
        if self.conf.include_adults:
            ret.append(self.split_cohort[DataSplit.TRAIN])
        return ret

    def test_data(self) -> Sequence:
        return self.split_cohort[DataSplit.TEST]

def parse_age(age: str) -> float:
    if pd.isna(age):
        return math.nan

    match age[-1].lower():
        case 'y':
            return float(age[:-1])
        case 'm':
            return float(age[:-1]) / 12
        case _:
            raise ValueError

def load_merged_plan():
    plan = pd.read_excel(PROCESSED_DIR / 'plan.xlsx', sheet_name='merge', dtype={MBDataKey.NUMBER: 'string'})
    plan.set_index(MBDataKey.NUMBER, inplace=True)
    assert plan.index.unique().size == plan.index.size
    return plan

def load_split() -> pd.Series:
    split = pd.read_excel(PROCESSED_DIR / 'split.xlsx', dtype={MBDataKey.NUMBER: 'string'})
    split.set_index(MBDataKey.NUMBER, inplace=True)
    return split['split']
