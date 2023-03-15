from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import itertools
import json
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import auc, roc_auc_score, roc_curve

import torch
from torch import nn
import torchmetrics
from torchmetrics.utilities.enums import AverageMethod

from mbs.model import MBModel
from mbs.utils.enums import MBDataKey
from umei.utils import DataKey, UMeIParser

from mbs.args import MBArgs
from mbs.datamodule import MBDataModule

@dataclass
class MBTestArgs(MBArgs):
    p_seeds: list[int] = field(default=None)
    p_seg_seeds: list[int] = field(default=None)
    val_cache_num: int = field(default=0)

    def __post_init__(self):
        super().__post_init__()
        if self.p_seg_seeds is None:
            self.p_seg_seeds = self.p_seeds

class MBTester(pl.LightningModule):
    models: nn.ModuleList | Sequence[Sequence[MBModel]]

    def __init__(self, args: MBTestArgs):
        super().__init__()
        self.args = args
        self.models = nn.ModuleList([
            nn.ModuleList()
            for _ in range(args.num_folds)
        ])
        for (seed, seg_seed), fold_id in itertools.product(zip(args.p_seeds, args.p_seg_seeds), range(args.num_folds)):
            seg_indicator = ''
            if args.seg_output_dir is not None:
                seg_indicator = f'seg-{seg_seed}'
            ckpt_path = self.args.output_dir / seg_indicator / f'run-{seed}' / f'fold-{fold_id}' / args.cls_scheme / 'last.ckpt'
            # ckpt_path = self.args.output_dir / f'run-{seed}' / f'fold-{fold_id}' / 'cls'
            # for filepath in ckpt_path.iterdir():
            #     if filepath.name.startswith('best'):
            #         ckpt_path = filepath
            #         break
            # else:
            #     raise RuntimeError
            self.models[fold_id].append(MBModel.load_from_checkpoint(ckpt_path, strict=True, args=args))
            print(f'load model from {ckpt_path}')

        self.metrics: Mapping[str, torchmetrics.Metric] = nn.ModuleDict({
            k: metric_cls(task='multiclass', num_classes=self.args.num_cls_classes, average=average)
            for k, metric_cls, average in [
                ('auroc', torchmetrics.AUROC, AverageMethod.NONE),
                ('recall', torchmetrics.Recall, AverageMethod.NONE),
                ('precision', torchmetrics.Precision, AverageMethod.NONE),
                ('f1', torchmetrics.F1Score, AverageMethod.NONE),
                ('acc', torchmetrics.Accuracy, AverageMethod.MICRO),
            ]
        })
        self.case_outputs = []
        self.test_report = {}
        self.val_fold = None

    def reset(self):
        for k, m in self.metrics.items():
            m.reset()
        self.case_outputs.clear()
        self.test_report = {}

    # def on_test_epoch_start(self) -> None:
    #     self.reset()

    def test_step(self, batch: dict, *args, **kwargs):
        prob = None
        if self.val_fold == 'test':
            models = list(itertools.chain(*self.models))
        else:
            models = self.models[self.val_fold]
        for model in models:
            logit = model.forward_cls(batch[DataKey.IMG], tta=self.args.do_tta)
            if prob is None:
                prob = logit.softmax(dim=-1)
            else:
                prob += logit.softmax(dim=-1)
        prob /= len(models)
        cls = batch[DataKey.CLS]
        for k, metric in self.metrics.items():
            metric(prob, cls)
        for i, case in enumerate(batch[MBDataKey.CASE]):
            self.case_outputs.append({
                'case': case,
                'true': self.args.cls_names[cls[i].item()],
                'pred': self.args.cls_names[prob[i].argmax().item()],
                **{
                    cls: prob[i, j].item()
                    for j, cls in enumerate(self.args.cls_names)
                },
                'prob': prob[i].cpu().numpy(),
            })

    def test_epoch_end(self, *args) -> None:
        self.test_report['n'] = len(self.case_outputs)
        self.test_report['seeds'] = self.args.p_seeds
        self.test_report['seg_seeds'] = self.args.p_seg_seeds
        for k, metric in self.metrics.items():
            m = metric.compute()
            if metric.average == AverageMethod.NONE:
                result = {}
                for i, sg in enumerate(self.args.cls_names):
                    result[sg] = m[i].item()
                result['macro avg'] = m.mean().item()
                self.test_report[k] = result
            else:
                self.test_report[k] = m.item()
        print(json.dumps(self.test_report, indent=4, ensure_ascii=False))

def plot_roc(y_true: np.ndarray, y_score: np.ndarray, target_names: list[str], output_dir: Path):
    target_names = deepcopy(target_names)
    y_true = np.eye(len(target_names))[y_true]
    # y_score = np.array(self.y_score)
    fig, ax = plt.subplots()
    lw = 2
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    if len(target_names) == 2 and target_names[0] == target_names[1]:
        target_names.pop()
    for i, name in enumerate(target_names):
        fpr, tpr, thresholds = roc_curve(y_true[:, i], y_score[:, i])
        ax.plot(fpr, tpr, lw=lw, label=f'{name}, AUC = %0.2f' % auc(fpr, tpr))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curves')
        ax.legend(loc="lower right")
    fig.savefig(output_dir / 'test-roc.pdf')
    fig.savefig(output_dir / 'test-roc.png')

def plot_roc_curve(y_test, y_score, model_name, output_dir):
    import seaborn as sns
    sns.set()
    fig = plt.figure(figsize=(7, 7))

    ns_preds = [0 for _ in range(len(y_test))]

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_preds)

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    plt.plot(fpr, tpr, marker='.', label=model_name)
    plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=16)
    plt.ylabel('Sensitivity (True Positive Rate)', fontsize=16)

    plt.legend(loc='lower right')
    plt.title(f'{model_name}: ROC Curve for Test Set', fontsize=20, fontweight="semibold")
    short_auc = round(auc, 4)
    plt.text(.93, .1, "AUC: " + str(short_auc),
             horizontalalignment="center", verticalalignment="center",
             fontsize=14, fontweight="semibold")
    print(233)
    fig.savefig(output_dir / 'test-roc-a.pdf')
    fig.savefig(output_dir / 'test-roc-a.png')

    # plt.show()

def run_test():
    (output_dir := args.output_dir / 'eval-test' / args.cls_scheme).mkdir(parents=True, exist_ok=True)
    tester.reset()
    tester.val_fold = 'test'
    trainer.test(tester, dataloaders=datamodule.test_dataloader())
    results_df = pd.DataFrame.from_records(tester.case_outputs)
    cls_names = args.cls_names
    y_true = np.array([
        args.cls_names.index(case['true'])
        for case in tester.case_outputs
    ])
    y_score = np.array([
        case['prob']
        for case in tester.case_outputs
    ])
    if len(cls_names) == 2:
        cls_names = [args.cls_scheme, args.cls_scheme]
    plot_roc(y_true, y_score, cls_names, output_dir)
    if len(cls_names) == 2:
        plot_roc_curve(y_true, y_score[:, 1], 'swt', output_dir)

    results_df.to_csv(output_dir / 'test-results.csv', index=False)
    results_df.to_excel(output_dir / 'test-results.xlsx', index=False)
    with open(output_dir / 'test-report.json', 'w') as f:
        json.dump(tester.test_report, f, indent=4, ensure_ascii=False)

def run_cv():
    (output_dir := args.output_dir / 'eval-cv' / args.cls_scheme).mkdir(parents=True, exist_ok=True)
    tester.reset()
    for i in range(args.num_folds):
        tester.val_fold = i
        datamodule.val_id = i
        trainer.test(tester, dataloaders=datamodule.val_dataloader(include_test=False))
    results_df = pd.DataFrame.from_records(tester.case_outputs)
    results_df.to_csv(output_dir / 'cv-results.csv', index=False)
    results_df.to_excel(output_dir / 'cv-results.xlsx', index=False)
    with open(output_dir / 'cv-report.json', 'w') as f:
        json.dump(tester.test_report, f, indent=4, ensure_ascii=False)

args: MBTestArgs
datamodule: MBDataModule
trainer: pl.Trainer
tester: MBTester

def main():
    global args, datamodule, trainer, tester

    parser = UMeIParser((MBTestArgs,), use_conf=True)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    datamodule = MBDataModule(args)
    tester = MBTester(args)
    trainer = pl.Trainer(
        logger=False,
        num_nodes=args.num_nodes,
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        precision=args.precision,
        benchmark=True,
    )

    run_test()
    run_cv()

if __name__ == '__main__':
    main()
