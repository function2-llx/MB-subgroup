from argparse import ArgumentParser, Namespace
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
from matplotlib.axes import Axes
from sklearn.metrics import roc_curve, auc
from torch import nn
from torchmetrics.classification import MulticlassAUROC
from torchmetrics.functional.classification import binary_accuracy
from torchmetrics.utilities.enums import AverageMethod

from luolib.models.lightning.cls_model import MetricsCollection

from mbs.utils.enums import MBGroup, MBDataKey, SUBGROUPS
from mbs.datamodule import load_merged_plan, load_split

args: Namespace
case_outputs: pd.DataFrame
plan = load_merged_plan()
split = load_split()
test_plan = plan[split == 'test']

def run(group: MBGroup | None = None):
    test_outputs = case_outputs[case_outputs['split'] == 'test']
    if group is not None:
        test_outputs = test_outputs[test_plan[MBDataKey.GROUP] == group]
    all_subgroups = test_outputs['true']
    from datasets import ClassLabel
    subgroups = ClassLabel(names=sorted(np.unique(all_subgroups).tolist(), key=lambda x: SUBGROUPS.index(x)))
    label = torch.tensor(test_outputs['true'].map(subgroups.str2int).to_numpy())
    pred = torch.tensor(test_outputs['pred'].map(subgroups.str2int).to_numpy())
    prob = torch.tensor(np.stack([test_outputs[subgroup].to_numpy() for subgroup in subgroups.names], axis=-1))
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

    group_name = 'all' if group is None else group
    output_dir: Path = args.output_dir / group_name
    output_dir.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame()
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
    report.to_excel(output_dir / 'report.xlsx')

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 12
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.set_facecolor('lightgray')
    for i, subgroup in enumerate(subgroups.names):
        subgroup_id = SUBGROUPS.index(subgroup)
        ax: Axes = axes[subgroup_id >> 1, subgroup_id & 1]
        ax.grid(visible=True, alpha=0.5, linestyle='--')
        fpr, tpr, _ = roc_curve(label == i, prob[:, i])
        ax.plot(fpr, tpr, color='#1CA9C9', lw=2, label=f'ROC curve')
        ax.fill_between(fpr, tpr, color='lightblue', label='Area under the curve (AUC)')
        ax.scatter(cx := 1 - report.at[subgroup, 'spe'], cy := report.at[subgroup, 'sen'], s=23,
                   marker='o', color='red', label='Current classifier', zorder=2)
        ax.annotate(f'({cx * 100:.1f}%, {cy * 100:.1f}%)', (cx, cy), textcoords="offset points", xytext=(33, -11), ha='center', fontsize=9)
        auroc = auc(fpr, tpr)
        assert auroc - report.at[subgroup, 'auroc'] < 1e-5
        ax.text(0.7, 0.5, f'{subgroup}\nAUC={auroc:.3f}', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='white'))
        ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(output_dir / f'roc.pdf')
    fig.savefig(output_dir / f'roc.png')

def main():
    global case_outputs, args
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('--output_dir', type=Path, default='plot')
    args = parser.parse_args()
    case_outputs = pd.read_excel(args.file, sheet_name='4way', dtype={'case': 'string'}).set_index('case')
    run()
    run(MBGroup.CHILD)
    run(MBGroup.ADULT)

if __name__ == '__main__':
    main()
