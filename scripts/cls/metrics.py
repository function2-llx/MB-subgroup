from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TypeAlias

import itertools as itt

from datasets import ClassLabel
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
import torch
from torch import nn
import torchmetrics
from torchmetrics.classification import MulticlassAUROC
from torchmetrics.functional.classification import binary_accuracy
from torchmetrics.utilities.enums import AverageMethod

from luolib.utils import process_map

from mbs.datamodule import load_merged_plan, load_split
from mbs.utils.enums import MBDataKey, MBGroup, SUBGROUPS

MetricsCollection: TypeAlias = Mapping[str, torchmetrics.Metric]

plan = load_merged_plan()
split = load_split()
test_plan = plan[split == 'test']
n_bootstraps = 10000

def compute(df: pd.DataFrame, subgroups: ClassLabel, seed: int | None = None):
    if seed is not None:
        rng = np.random.default_rng(seed)
        groups = [
            group.sample(frac=1, random_state=rng, replace=True)
            for _, group in df.groupby('true')
        ]
        df = pd.concat(groups)
    label = torch.tensor(df['true'].map(subgroups.str2int).to_numpy())
    pred = torch.tensor(df['pred'].map(subgroups.str2int).to_numpy())
    prob = torch.tensor(np.stack([df[subgroup].to_numpy() for subgroup in subgroups.names], axis=-1))
    report = pd.DataFrame()
    metrics: MetricsCollection = nn.ModuleDict({
        k: metric_cls(task='multiclass', num_classes=subgroups.num_classes, average=AverageMethod.NONE)
        for k, metric_cls in [
            ('f1', torchmetrics.F1Score),
            ('sen', torchmetrics.Recall),
            ('spe', torchmetrics.Specificity),
            ('acc', torchmetrics.Accuracy),
            ('auroc', torchmetrics.AUROC),
        ]
    })
    for name, metric in metrics.items():
        m = metric(prob, label).numpy()
        for i, subgroup in enumerate(subgroups.names):
            if name == 'acc':
                m[i] = binary_accuracy(pred == i, label == i).item()
            report.at[subgroup, name] = m[i]
        report.at['macro', name] = m.mean()
        metric.average = AverageMethod.MICRO
        metric._computed = None
        if not isinstance(metric, MulticlassAUROC):
            report.at['micro', name] = metric.compute().item()
    return report

def run(config_name: str, output_root: Path, split: str, group: MBGroup | None = None):
    output_dir = output_root / split / config_name
    df = pd.read_excel(
        f'MB-data/outputs/{split}/{config_name}.xlsx',
        dtype={'case': pd.StringDtype()},
    ).set_index('case')
    if group is not None:
        df = df[df['group'] == group]
        output_dir = output_dir / group
    all_subgroups = df['true']
    subgroups = ClassLabel(names=sorted(np.unique(all_subgroups).tolist(), key=lambda x: SUBGROUPS.index(x)))

    output_dir.mkdir(parents=True, exist_ok=True)

    report = compute(df, subgroups)
    report.to_excel(output_dir / 'report.xlsx')
    bs_reports = process_map(
        partial(compute, df, subgroups),
        np.random.SeedSequence(42).generate_state(n_bootstraps).tolist(),
        max_workers=8,
        chunksize=50,
        dynamic_ncols=True,
    )
    bs_metrics = np.stack([bs_report.to_numpy() for bs_report in bs_reports])
    np.save(output_dir / 'bs_metrics.npy', bs_metrics)
    ci = np.stack(
        [
            np.percentile(bs_metrics, q=2.5, axis=0),
            np.percentile(bs_metrics, q=97.5, axis=0),
        ],
        axis=-1,
    )
    ci_report = pd.DataFrame()
    for cls_idx, metric_idx in np.ndindex(report.shape):
        cls_name = report.index[cls_idx]
        metric_name = report.columns[metric_idx]
        if pd.isna(mean := report.at[cls_name, metric_name]):
            continue
        ci_report.at[cls_name, metric_name] = mean
        for ci_idx in range(2):
            ci_report.at[cls_name, f'{metric_name}-{ci_idx}'] = ci[cls_idx, metric_idx, ci_idx]
    ci_report.to_excel(output_dir / 'report-ci.xlsx')

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    prob = torch.tensor(np.stack([df[subgroup].to_numpy() for subgroup in subgroups.names], axis=-1))
    label = torch.tensor(df['true'].map(subgroups.str2int).to_numpy())
    fig.set_facecolor('lightgray')
    for i, subgroup in enumerate(subgroups.names):
        subgroup_id = SUBGROUPS.index(subgroup)
        # if subgroups.num_classes == 4:
        #     ax: Axes = axes[subgroup_id >> 1, subgroup_id & 1]
        # else:
        #     # sorry
        #     ax: Axes = axes[subgroup_ids.index(subgroup_id)]
        ax.grid(visible=True, alpha=0.5, linestyle='--')
        fpr, tpr, _ = roc_curve(label == i, prob[:, i])
        ax.plot(fpr, tpr, color='#1CA9C9', lw=2, label=f'ROC curve')
        ax.fill_between(fpr, tpr, color='lightblue', label='Area under the curve (AUC)')
        # ax.scatter(cx := 1 - report.at[subgroup, 'spe'], cy := report.at[subgroup, 'sen'], s=23,
        #            marker='o', color='red', label='Current classifier', zorder=2)
        # ax.annotate(f'({cx * 100:.1f}%, {cy * 100:.1f}%)', (cx, cy), textcoords="offset points", xytext=(33, -11), ha='center', fontsize=9)
        auroc = auc(fpr, tpr)
        assert auroc - report.at[subgroup, 'auroc'] < 1e-5
        ax.text(0.7, 0.5, f'{subgroup}\nAUC={auroc:.3f}', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='white'))
        ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(output_dir / f'roc.pdf')
    fig.savefig(output_dir / f'roc.png')

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_root', type=Path, default='MB-data/metrics')
    args = parser.parse_args()
    for config_name, split in itt.product(
        ['mbmf', 'scratch', 'radiomics'],
        ['test', 'val'],
    ):
        run(config_name, args.output_root, split)
        if config_name == 'mbmf' and split == 'test':
            run(config_name, args.output_root, split, MBGroup.CHILD)
            run(config_name, args.output_root, split, MBGroup.ADULT)

if __name__ == '__main__':
    main()
