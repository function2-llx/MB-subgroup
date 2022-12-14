from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import itertools
import json

import pandas as pd
import pytorch_lightning as pl
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

class MBTester(pl.LightningModule):
    models: nn.ModuleList | Sequence[Sequence[MBModel]]

    def __init__(self, args: MBTestArgs):
        super().__init__()
        self.args = args
        self.models = nn.ModuleList([
            nn.ModuleList()
            for _ in range(args.num_folds)
        ])
        for seed, fold_id in itertools.product(args.p_seeds, range(args.num_folds)):
            ckpt_path = self.args.output_dir / f'run-{seed}' / f'fold-{fold_id}' / args.cls_scheme / 'last.ckpt'
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
            k: metric_cls(num_classes=self.args.num_cls_classes, average=average)
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

def run_test():
    (output_dir := args.output_dir / args.cls_scheme).mkdir(parents=True, exist_ok=True)
    tester.reset()
    tester.val_fold = 'test'
    trainer.test(tester, dataloaders=datamodule.test_dataloader())
    results_df = pd.DataFrame.from_records(tester.case_outputs)
    results_df.to_csv(output_dir / 'test-results.csv', index=False)
    results_df.to_excel(output_dir / 'test-results.xlsx', index=False)
    with open(output_dir / 'test-report.json', 'w') as f:
        json.dump(tester.test_report, f, indent=4, ensure_ascii=False)

def run_cv():
    (output_dir := args.output_dir / args.cls_scheme).mkdir(parents=True, exist_ok=True)
    tester.reset()
    for i in range(args.num_folds):
        tester.val_fold = i
        datamodule.val_id = i
        trainer.test(tester, dataloaders=datamodule.val_dataloader())
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
