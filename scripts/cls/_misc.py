from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from luolib.types import PathLike
from luolib.utils import process_map

def auc_with_sample(y_true: np.ndarray, y_score: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    values: dict[..., list[int]] = {}
    for i, v in enumerate(y_true.tolist()):
        values.setdefault(v, []).append(i)
    idx = np.empty(y_true.shape, dtype=np.int64)
    num_cls = len(values)
    n = y_true.shape[0]
    idx[:-num_cls] = rng.integers(n, size=n - num_cls)
    for i, cls_idx in enumerate(values.values()):
        idx[-i - 1] = rng.choice(cls_idx)
    y_true = y_true[idx]
    y_score = y_score[idx]
    value = roc_auc_score(y_true, y_score, multi_class='ovr')
    return value

def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_bootstraps: int = 10000, seed: int = 42) -> tuple[float, float]:
    values = process_map(
        partial(auc_with_sample, y_true, y_score),
        np.random.SeedSequence(seed).generate_state(n_bootstraps).tolist(),
        max_workers=16,
        chunksize=10,
    )
    return (
        np.percentile(values, q=2.5, axis=0),
        np.percentile(values, q=97.5, axis=0),
    )

def read_case_outputs(filepath: PathLike, split: Literal['train', 'test'] | None = 'test'):
    df = pd.read_excel(filepath, sheet_name='4way', dtype={'case': 'string'}).set_index('case')
    if split is not None:
        df = df[df['split'] == split]
    return df
