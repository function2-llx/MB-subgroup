from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import itertools
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import auc, roc_auc_score, roc_curve
import torch
from torch import nn

from luolib.conf import parse_exp_conf
from luolib.models import ClsModel
from luolib.utils import DataKey, DataSplit

from mbs.conf import MBClsConf
from mbs.models import MBClsModel
from mbs.datamodule import MBClsDataModule, load_split
from mbs.models.lightning.cls_model import get_cls_names

@dataclass(kw_only=True)
class MBClsTestConf(MBClsConf):
    p_seeds: list[int]
    include_adults: bool = False
    eval_batch_size: int = 1
    print_shape: bool = False
    val_cache_num: int = 0

class MBTester(ClsModel):
    conf: MBClsTestConf
    models: Sequence[Sequence[MBClsModel]]

    @property
    def cls_names(self):
        return get_cls_names(self.conf.cls_scheme)

    def __init__(self, conf: MBClsTestConf):
        super().__init__(conf)
        wuhu = self.create_metrics(conf.num_cls_classes)
        self.micro_metrics = wuhu
        self.models = nn.ModuleList([
            nn.ModuleList()
            for _ in range(conf.num_folds)
        ])
        for seed, fold_id in itertools.product(conf.p_seeds, range(conf.num_folds)):
            ckpt_path = self.conf.output_dir / f'run-{seed}' / f'fold-{fold_id}' / 'last.ckpt'
            self.models[fold_id].append(MBClsModel.load_from_checkpoint(ckpt_path, strict=True, conf=conf))
            print(f'load model from {ckpt_path}')

        self.case_outputs = []
        self.split = load_split()

    def test_step(self, batch: dict, *args, **kwargs):
        prob = None
        sample_case = batch[DataKey.CASE][0]
        split = self.split[sample_case]
        for case in batch[DataKey.CASE][1:]:
            assert split == self.split[case]

        if split == 'test':
            models = list(itertools.chain(*self.models))
        else:
            models = self.models[split]
        for model in models:
            logit = model.cal_logit(batch, flip=self.conf.do_tta)
            if prob is None:
                prob = logit.softmax(dim=-1)
            else:
                prob += logit.softmax(dim=-1)
        prob /= len(models)
        label = batch[DataKey.CLS]
        self.accumulate_metrics(prob, label, self.cls_metrics[DataSplit.TEST])
        if split != 'test':
            self.accumulate_metrics(prob, label, self.micro_metrics)
        for i, case in enumerate(batch[DataKey.CASE]):
            self.case_outputs.append({
                'case': case,
                'split': split,
                'true': self.cls_names[label[i].item()],
                'pred': self.cls_names[prob[i].argmax().item()],
                **{
                    label: prob[i, j].item()
                    for j, label in enumerate(self.cls_names)
                },
                'prob': prob[i].cpu().numpy(),
            })

    def plot_roc(self, save_dir: Path):
        conf = self.conf
        cls_names = self.cls_names
        y_true = np.array([
            self.cls_names.index(case['true'])
            for case in self.case_outputs if case['split'] == 'test'
        ])
        y_score = np.array([
            case['prob']
            for case in self.case_outputs if case['split'] == 'test'
        ])
        plot_roc(y_true, y_score, cls_names, save_dir)


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

# def run_test():
#     (output_dir := conf.output_dir / 'eval-test' / conf.cls_scheme).mkdir(parents=True, exist_ok=True)
#     tester.reset()
#     tester.val_fold = 'test'
#     trainer.test(tester, dataloaders=datamodule.test_dataloader())
#     results_df = pd.DataFrame.from_records(tester.case_outputs)
#     cls_names = conf.cls_names
#     y_true = np.array([
#         conf.cls_names.index(case['true'])
#         for case in tester.case_outputs
#     ])
#     y_score = np.array([
#         case['prob']
#         for case in tester.case_outputs
#     ])
#     if len(cls_names) == 2:
#         cls_names = [conf.cls_scheme, conf.cls_scheme]
#     plot_roc(y_true, y_score, cls_names, output_dir)
#     if len(cls_names) == 2:
#         plot_roc_curve(y_true, y_score[:, 1], 'swt', output_dir)

#     results_df.to_csv(output_dir / 'test-results.csv', index=False)
#     results_df.to_excel(output_dir / 'test-results.xlsx', index=False)
#     with open(output_dir / 'test-report.json', 'w') as f:
#         json.dump(tester.test_report, f, indent=4, ensure_ascii=False)

class MBClsTestDataModule(MBClsDataModule):
    def val_dataloader(self):
        return super(MBClsDataModule, self).val_dataloader()

def main():
    conf = parse_exp_conf(MBClsTestConf)
    datamodule = MBClsTestDataModule(conf)
    model = MBTester(conf)
    trainer = pl.Trainer(
        logger=False,
        accelerator='gpu',
        devices=torch.cuda.device_count(),
    )
    reports = {}
    for i in range(conf.num_folds):
        datamodule.val_id = i
        trainer.test(model, datamodule.val_dataloader())
        print(f'evaluating {i}')
        reports[f'fold-{i}'] = model.metrics_report(model.cls_metrics_test, model.cls_names)
    reports['macro'] = pd.DataFrame(reports).T.mean().to_dict()
    reports['micro'] = model.metrics_report(model.micro_metrics, model.cls_names)
    trainer.test(model, datamodule.test_dataloader())
    reports['test'] = model.metrics_report(model.cls_metrics_test, model.cls_names)
    
    save_dir = conf.output_dir / 'eval'
    save_dir.mkdir(exist_ok=True)
    pd.DataFrame(model.case_outputs).to_excel(save_dir / 'case-outputs.xlsx', freeze_panes=(1, 0))
    model.plot_roc(save_dir)
    pd.DataFrame(reports).to_excel(save_dir / 'report.xlsx', freeze_panes=(1, 0))

if __name__ == '__main__':
    main()
