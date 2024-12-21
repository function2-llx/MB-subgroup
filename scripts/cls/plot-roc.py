from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

import cytoolz
from datasets import ClassLabel
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve
import torchmetrics

from luolib.types import PathLike
from mbs.datamodule import load_merged_plan, load_split
from mbs.utils.enums import MBDataKey, MBGroup, SUBGROUPS
from _misc import bootstrap_auc_ci

MetricsCollection: TypeAlias = Mapping[str, torchmetrics.Metric]

plan = load_merged_plan()
split = load_split()
test_plan = plan[split == 'test']
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12

def plot_group(ax: Axes, df: pd.DataFrame, group: MBGroup | None = None):
    df_index = df['split'] == 'test'
    if group is not None:
        df_index &= test_plan[MBDataKey.GROUP] == group
    df = df[df_index]
    all_subgroups = df['true']
    subgroups = ClassLabel(names=sorted(np.unique(all_subgroups).tolist(), key=lambda x: SUBGROUPS.index(x)))
    colors = {
        'WNT': 'tab:blue',
        'SHH': 'tab:orange',
        'G3': 'tab:green',
        'G4': 'tab:red',
    }
    group_name = 'all' if group is None else group
    if group_name == 'child':
        group_name = 'pediatric'
    ax.set_title(group_name.capitalize())
    prob = np.stack([df[subgroup].to_numpy() for subgroup in subgroups.names], axis=-1)
    if group == 'adult':
        prob /= prob.sum(axis=1, keepdims=True)
    label = df['true'].map(subgroups.str2int).to_numpy()
    ax.grid(visible=True, alpha=0.5, linestyle='--')
    ax.plot([0, 1], linestyle='--', color='tab:purple')
    subgroup_aucs = np.empty(4, dtype=np.float64)
    for i, subgroup in enumerate(subgroups.names):
        fpr, tpr, _ = roc_curve(label == i, prob[:, i])
        subgroup_aucs[i] = auc(fpr, tpr)
        ci = bootstrap_auc_ci(label == i, prob[:, i])
        ax.plot(
            fpr, tpr,
            lw=2,
            label=f'{subgroup}, AUC={subgroup_aucs[i]:.3f} ({ci[0]:.3f}, {ci[1]:.3f})',
            color=colors[subgroup],
        )
    average_auc = subgroup_aucs.mean()
    ci = bootstrap_auc_ci(label, prob)
    ax.text(
        0.43, 0.25,
        f'Average AUC={average_auc:.3f} ({ci[0]:.3f}, {ci[1]:.3f})',
        bbox=dict(boxstyle='round, pad=0.5', facecolor='lightgray', alpha=0.5),
    )
    ax.legend(loc='lower right')
output_dir = Path('ROC-plot')
output_dir.mkdir(exist_ok=True, parents=True)

def read_case_output(path: PathLike):
    df = pd.read_excel(path, sheet_name='4way', dtype={'case': 'string'}).set_index('case')
    return df

def plot_list():
    case_output = read_case_output('case-outputs/dl.xlsx')
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.5))
    fig.tight_layout()
    fig.set_facecolor('lightgray')
    for ax, group in zip(axes, (None, MBGroup.CHILD, MBGroup.ADULT)):
        plot_group(ax, case_output, group)
    fig.savefig(output_dir / 'list.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    fig.savefig(output_dir / 'list.pdf', bbox_inches='tight', pad_inches=0.05)

def plot_cmp_subgroup(ax: Axes, subgroup: str):
    for task, name in [
        ('dl', 'DL'),
        ('dl-scratch', 'DL w/o seg.'),
        ('radiomics', 'Radiomics'),
    ]:
        df = read_case_output(f'case-outputs/{task}.xlsx')
        df_index = df['split'] == 'test'
        df = df[df_index]
        prob = df[subgroup].to_numpy()
        label = df['true'].to_numpy() == subgroup
        fpr, tpr, _ = roc_curve(label, prob)
        auroc = auc(fpr, tpr)
        ci = bootstrap_auc_ci(label, prob)
        ax.plot(fpr, tpr, lw=2, label=f'{name}, AUC={auroc:.3f} ({ci[0]:.3f}, {ci[1]:.3f})')

def plot_cmp():
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.tight_layout()
    fig.set_facecolor('lightgray')
    for subgroup, ax in zip(SUBGROUPS, cytoolz.concat(axes)):
        ax.set_title(subgroup)
        ax.grid(visible=True, alpha=0.5, linestyle='--')
        ax.plot([0, 1], linestyle='--', color='tab:purple')
        plot_cmp_subgroup(ax, subgroup)
        ax.legend(loc='lower right')
    fig.savefig(output_dir / 'cmp.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    fig.savefig(output_dir / 'cmp.pdf', bbox_inches='tight', pad_inches=0.05)

def compute_others():
    for task, name in [
        ('dl', 'DL'),
        ('dl-scratch', 'DL w/o seg.'),
        ('radiomics', 'Radiomics'),
    ]:
        df = read_case_output(f'case-outputs/{task}.xlsx')
        df_index = df['split'] == 'test'
        df = df[df_index]
        prob = np.stack([df[subgroup].to_numpy() for subgroup in SUBGROUPS], axis=-1)
        label = df['true'].map(SUBGROUPS.index).to_numpy()
        value = roc_auc_score(label, prob, multi_class='ovr')
        ci = bootstrap_auc_ci(label, prob)
        print(task)
        print(value, *ci)

def main():
    # plot_list()
    plot_cmp()
    # compute_others()

if __name__ == '__main__':
    main()
