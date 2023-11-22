from collections.abc import Hashable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence

import math
import numpy as np
import pandas as pd
import torch
from torch.types import Device

from luolib.datamodule import CrossValDataModule
from luolib.nnunet import nnUNet_preprocessed
from luolib.types import tuple2_t, tuple3_t
from monai.utils import GridSampleMode

from mbs.utils.enums import DATA_DIR, MBDataKey, MBGroup, PROCESSED_DIR

__all__ = [
    'MBDataModuleBase',
    'load_clinical',
    'load_merged_plan',
    'load_split',
    'parse_age',
]

def load_clinical():
    clinical = pd.read_excel(DATA_DIR / '影像预测分子分型.xlsx', dtype={'number': 'string'}).set_index('number')
    clinical.rename(columns={'gender': 'sex'}, inplace=True)
    return clinical

class MBDataModuleBase(CrossValDataModule):
    def __init__(
        self,
        *args,
        data_dir: Path = nnUNet_preprocessed / 'Dataset500_TTMB' / 'nnUNetPlans-z_3d_fullres',
        include_adults: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.include_adults = include_adults

    @cached_property
    def plan(self):
        plan = load_merged_plan()
        if not self.include_adults:
            plan = plan[plan[MBDataKey.GROUP] == MBGroup.CHILD]
        return plan

    @cached_property
    def case_split(self):
        return load_split()

    @cached_property
    def clinical(self):
        return load_clinical()

    def extract_case_data(self, number: str):
        age: int = self.clinical.at[number, 'age']
        sex: int = self.clinical.at[number, 'sex']
        clinical_vec = torch.zeros(3)
        clinical_vec[0] = age / 100
        clinical_vec[sex] = 1
        case_data = {
            'case': number,
            MBDataKey.CLINICAL: clinical_vec,
            MBDataKey.SUBGROUP: self.plan.at[number, 'subgroup'],
        }
        return case_data

    def fit_data(self) -> dict[str, dict]:
        number: str  # make PyCharm calm
        return {
            number: self.extract_case_data(number)
            for number in self.plan.index if self.case_split[number] != 'test'
        }

    def splits(self) -> Sequence[tuple[Sequence[Hashable], Sequence[Hashable]]]:
        fit_case_split = self.case_split[self.case_split != 'test']
        ret = []
        for val_name in fit_case_split.unique():
            train = []
            val = []
            for case, split in fit_case_split.items():
                if split == val_name:
                    val.append(case)
                else:
                    train.append(case)
            ret.append((train, val))
        return ret

    def test_data(self):
        return [
            self.extract_case_data(number)
            for number in self.plan.index if self.case_split[number] == 'test'
        ]

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
    """
    Returns:
        map of case number → split (val fold id / test)
    """
    split = pd.read_excel(PROCESSED_DIR / 'split.xlsx', dtype={MBDataKey.NUMBER: 'string'})
    split.set_index(MBDataKey.NUMBER, inplace=True)
    return split['split']

@dataclass(kw_only=True)
class TransConfBase:
    patch_size: tuple3_t[int]
    device: Device = 'cpu'

    @dataclass
    class Scale:
        prob: float = 0.2
        range: tuple2_t[float] = (0.7, 1.4)
        ignore_dim: int | None = 0

    scale: Scale

    @dataclass
    class Rotate:
        prob: float = 0.2
        range: tuple3_t[float] = (np.pi / 2, 0, 0)

    rotate: Rotate

    @dataclass
    class GaussianNoise:
        prob: float = 0.1
        max_std: float = 0.1

    gaussian_noise: GaussianNoise

    @dataclass
    class GaussianSmooth:
        prob: float = 0.2
        prob_per_channel: float = 0.5
        sigma_x: tuple2_t[float] = (0.5, 1)
        sigma_y: tuple2_t[float] = (0.5, 1)
        sigma_z: tuple2_t[float] = (0.5, 1)

    gaussian_smooth: GaussianSmooth

    @dataclass
    class ScaleIntensity:
        prob: float = 0.15
        range: tuple2_t[float] = (0.75, 1.25)

        @property
        def factors(self):
            return self.range[0] - 1, self.range[1] - 1

    scale_intensity: ScaleIntensity

    @dataclass
    class AdjustContrast:
        prob: float = 0.15
        range: tuple2_t[float] = (0.75, 1.25)
        preserve_intensity_range: bool = True

    adjust_contrast: AdjustContrast

    @dataclass
    class SimulateLowResolution:
        prob: float = 0.25
        prob_per_channel: float = 0.5
        zoom_range: tuple2_t[float] = (0.5, 1)
        downsample_mode: str | int = GridSampleMode.NEAREST
        upsample_mode: str | int = GridSampleMode.BICUBIC

    simulate_low_resolution: SimulateLowResolution

    @dataclass
    class GammaCorrection:
        prob: float = 0.3
        range: tuple2_t[float] = (0.7, 1.5)
        prob_invert: float = 0.75
        retain_stats: bool = True

    gamma_correction: GammaCorrection
