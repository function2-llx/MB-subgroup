from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import itertools
from pathlib import Path

import torchmetrics
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import auc, roc_auc_score, roc_curve
import torch
from torch import nn
from torchmetrics.utilities.enums import AverageMethod

from luolib.conf import parse_exp_conf, parse_cli
from luolib.models.lightning.cls_model import MetricsCollection
from luolib.utils import DataKey

from mbs.conf import MBClsConf, get_cls_map_vec, get_cls_names
from mbs.datamodule import MBClsDataModule, load_split
from mbs.models import MBClsModel

@dataclass(kw_only=True)
class MBClsTestConf:
    exp_conf_cls: str
    exp_conf_path: Path
    exp_conf: MBClsConf
    datamodule_cls: str
    inferer_cls: str

    p_seeds: list[int]
    # include_adults: bool = False
    # eval_batch_size: int = 1
    # print_shape: bool = False
    # val_cache_num: int = 0
    vote: bool = False

    def load_exp_conf(self):
        # assert self.exp_conf_cls in [MBM2FConf.__name__, MBSegConf.__name__]
        import mbs.conf
        exp_conf_cls = getattr(mbs.conf, self.exp_conf_cls)
        self.exp_conf = OmegaConf.merge(    # type: ignore
            parse_exp_conf(exp_conf_cls, self.exp_conf_path),
            # make omegaconf happy, or it may complain that the class of `conf.exp_conf` is not subclass of `exp_conf_cls`
            OmegaConf.to_container(self.exp_conf),
        )

class Scheme(nn.Module):
    def __init__(self, scheme: str):
        super().__init__()
        self.scheme = scheme
        self.names = get_cls_names(scheme)
        self.map_vec = get_cls_map_vec(scheme)
        self.metrics = MBTester.create_metrics(self.num_classes)
        self.micro_metrics = MBTester.create_metrics(self.num_classes)
        self.report = {}
        self.case_output = []

    @property
    def num_classes(self):
        return len(self.names)

    def accumulate(self, prob: torch.Tensor, label: torch.Tensor, pred: torch.Tensor, split: str, cases: list[str]):
        batch_size = prob.shape[0]
        cls_map_vec = get_cls_map_vec(self.scheme, prob.device)
        scheme_label = cls_map_vec[label]
        scheme_pred = cls_map_vec[pred]
        scheme_prob = torch.zeros(batch_size, self.num_classes, device=prob.device)
        scheme_prob.scatter_add_(1, cls_map_vec.expand(batch_size, -1), prob)
        MBClsModel.accumulate_metrics(scheme_prob, scheme_label, self.metrics, scheme_pred)
        if split != 'test':
            MBClsModel.accumulate_metrics(scheme_prob, scheme_label, self.micro_metrics, scheme_pred)

        for i, case in enumerate(cases):
            self.case_output.append(
                {
                    'case': case,
                    'split': split,
                    'true': self.names[scheme_label[i].item()],
                    'pred': self.names[scheme_pred[i].item()],
                    **{
                        label: scheme_prob[i, j].item()
                        for j, label in enumerate(self.names)
                    },
                    'prob': scheme_prob[i].cpu().numpy(),
                }
            )

class MBTester(pl.LightningModule):
    conf: MBClsTestConf
    models: Sequence[Sequence[MBClsModel]] | nn.ModuleList
    schemes: Sequence[Scheme] | nn.ModuleList

    @property
    def exp_conf(self):
        return self.conf.exp_conf

    @staticmethod
    def create_metrics(num_classes: int) -> MetricsCollection:
        return nn.ModuleDict({
            k: (
                metric := metric_cls(task='multiclass', num_classes=num_classes, average=average),
                setattr(metric, '_luolib_force_prob', force_prob),
                metric,
            )[-1]
            for k, metric_cls, average, force_prob in [
                ('sen', torchmetrics.Recall, AverageMethod.NONE, False),
                ('spe', torchmetrics.Specificity, AverageMethod.NONE, False),
                ('auroc', torchmetrics.AUROC, AverageMethod.NONE, True),
                ('f1', torchmetrics.F1Score, AverageMethod.NONE, False),
                ('acc', torchmetrics.Accuracy, AverageMethod.MICRO, False),
            ]
        })

    def __init__(self, conf: MBClsTestConf):
        super().__init__()
        self.conf = conf
        exp_conf = self.exp_conf
        import mbs.models
        model_cls: type[MBClsModel] = getattr(mbs.models, conf.inferer_cls)
        self.schemes = nn.ModuleList([Scheme(scheme) for scheme in ['4way', '3way', 'WS-G34']])
        self.models = nn.ModuleList([
            nn.ModuleList()
            for _ in range(exp_conf.num_folds)
        ])
        for seed, fold_id in itertools.product(conf.p_seeds, range(exp_conf.num_folds)):
            ckpt_path = exp_conf.output_dir / f'run-{seed}' / f'fold-{fold_id}' / 'last.ckpt'
            self.models[fold_id].append(model_cls.load_from_checkpoint(ckpt_path, strict=True, conf=exp_conf))
            print(f'load model from {ckpt_path}')

        self.split = load_split()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        for scheme in self.schemes:
            for metric in scheme.metrics.values():
                metric.reset()

    def test_step(self, batch: dict, *args, **kwargs):
        conf = self.conf
        exp_conf = conf.exp_conf
        prob = None
        cases = batch[DataKey.CASE]
        batch_size = len(cases)
        sample_case = cases[0]
        split = self.split[sample_case]
        for case in batch[DataKey.CASE][1:]:
            assert split == self.split[case]

        if split == 'test':
            models = list(itertools.chain(*self.models))
        else:
            models = self.models[split]
        pred = torch.empty(batch_size, len(models), dtype=torch.int32, device=self.device)
        for i, model in enumerate(models):
            logit = model.infer_logit(batch)
            cur_prob = logit.softmax(dim=-1)
            pred[:, i] = cur_prob.argmax(dim=-1)
            if prob is None:
                prob = cur_prob
            else:
                prob += cur_prob
        prob /= len(models)
        vote = torch.empty(batch_size, exp_conf.num_cls_classes, dtype=torch.int32, device=self.device)
        for i in range(batch_size):
            vote[i] = pred[i].bincount(minlength=exp_conf.num_cls_classes)
        max_vote, vote_pred = vote.max(dim=-1, keepdim=True)
        pred = prob.argmax(dim=-1)
        if conf.vote:
            pred = torch.where((vote == max_vote).sum(dim=-1) == 1, vote_pred[:, 0], pred)
        label = batch[DataKey.CLS]
        for scheme in self.schemes:
            scheme.accumulate(prob, label, pred, split, batch[DataKey.CASE])

    def plot_roc(self, save_dir: Path):
        for scheme in self.schemes:
            y_true = np.array([
                scheme.names.index(case['true'])
                for case in scheme.case_output if case['split'] == 'test'
            ])
            y_score = np.array([
                case['prob']
                for case in scheme.case_output if case['split'] == 'test'
            ])
            plot_roc(y_true, y_score, scheme.names, save_dir / 'roc', scheme.scheme)

def plot_roc(y_true: np.ndarray, y_score: np.ndarray, target_names: list[str], output_dir: Path, save_name: str):
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
    output_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_dir / f'{save_name}.pdf')
    fig.savefig(output_dir / f'{save_name}.png')

# copy from radiomics
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

# class MBClsTestDataModule(MBClsDataModule):
#     def val_dataloader(self):
#         return super(MBClsDataModule, self).val_dataloader()

def main():
    # conf = parse_exp_conf(MBClsTestConf)
    conf, _ = parse_cli(MBClsTestConf)
    MBClsTestConf.load_exp_conf(conf)
    torch.set_float32_matmul_precision(conf.exp_conf.float32_matmul_precision)
    import mbs.datamodule
    datamodule_cls = getattr(mbs.datamodule, conf.datamodule_cls)
    datamodule = datamodule_cls(conf.exp_conf)
    tester = MBTester(conf)
    trainer = pl.Trainer(
        logger=False,
        accelerator='gpu',
        devices=torch.cuda.device_count(),
    )
    for i in range(conf.exp_conf.num_folds):
        datamodule.val_id = i
        trainer.test(tester, datamodule.val_dataloader())
        print(f'evaluating {i}')
        for scheme in tester.schemes:
            scheme.report[f'fold-{i}'] = MBClsModel.metrics_report(scheme.metrics, scheme.names)
    for scheme in tester.schemes:
        scheme.report['macro'] = pd.DataFrame(scheme.report).T.mean().to_dict()
        scheme.report['micro'] = MBClsModel.metrics_report(scheme.micro_metrics, scheme.names)
    trainer.test(tester, datamodule.test_dataloader())
    for scheme in tester.schemes:
        scheme.report['test'] = MBClsModel.metrics_report(scheme.metrics, scheme.names)

    save_dir = conf.exp_conf.output_dir / f"eval-{'+'.join(map(str, sorted(conf.p_seeds)))}{'-tta' if conf.exp_conf.do_tta else ''}"
    save_dir.mkdir(exist_ok=True)
    tester.plot_roc(save_dir)
    with pd.ExcelWriter(save_dir / f'case-output.xlsx') as case_output_writer, pd.ExcelWriter(save_dir / 'report.xlsx') as report_writer:
        for scheme in tester.schemes:
            pd.DataFrame(scheme.case_output).to_excel(case_output_writer, scheme.scheme, freeze_panes=(1, 1), index=False)
            pd.DataFrame(scheme.report).to_excel(report_writer, scheme.scheme, freeze_panes=(1, 1))

if __name__ == '__main__':
    main()
