from collections.abc import Hashable, Mapping
from functools import cached_property
import math
from typing import Sequence

import pandas as pd
import torch

from luolib.datamodule import CrossValDataModule
from luolib.datamodule.base import DataSeq
from luolib.utils import DataSplit, DataKey

# from mbs.conf import MBConfBase
from mbs.utils.enums import CLINICAL_DIR, MBDataKey, MBGroup, PROCESSED_DIR

def load_clinical():
    clinical = pd.read_excel(CLINICAL_DIR / '影像预测分子分型.xlsx', dtype={'number': 'string'}).set_index('number')
    clinical.rename(columns={'gender': 'sex'}, inplace=True)
    return clinical

class MBDataModuleBase(CrossValDataModule):
    # conf: MBConfBase

    @cached_property
    def split_cohort(self) -> dict[Hashable, DataSeq]:
        plan = load_merged_plan()
        split = load_split()
        clinical = load_clinical()
        if not self.conf.include_adults:
            plan = plan[plan[MBDataKey.GROUP] == MBGroup.CHILD]
        split_cohort = {}
        for number, info in plan.iterrows():
            case_data_dir = self.conf.data_dir / number
            age: int = clinical.at[number, 'age']
            sex: int = clinical.at[number, 'sex']
            clinical_vec = torch.zeros(3)
            clinical_vec[0] = age / 100
            clinical_vec[sex] = 1
            split_cohort.setdefault(split[number], []).append({
                DataKey.CASE: number,
                **{
                    key: path
                    for key in [DataKey.IMG, DataKey.SEG] if (path := case_data_dir / f'{key}.npy').exists()
                },
                MBDataKey.CLINICAL: clinical_vec,
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
        # assert self.conf.include_adults
        # if self.conf.include_adults:
        #     ret.append(self.split_cohort[DataSplit.TRAIN])
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
    plan.sort_index(inplace=True)
    assert plan.index.unique().size == plan.index.size
    return plan

def load_split() -> pd.Series:
    split = pd.read_excel(PROCESSED_DIR / 'split.xlsx', dtype={MBDataKey.NUMBER: 'string'})
    split.set_index(MBDataKey.NUMBER, inplace=True)
    return split['split']
