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
from mbs.utils.enums import MBDataKey, SUBGROUPS
from umei.utils import DataKey, UMeIParser

from mbs.args import MBArgs
from mbs.datamodule import MBDataModule

@dataclass
class MBTestArgs(MBArgs):
    p_seeds: list[int] = field(default=None)

class MBTester(pl.LightningModule):
    models: nn.ModuleList | Sequence[MBModel]

    def __init__(self, args: MBTestArgs):
        super().__init__()
        self.args = args
        self.models = nn.ModuleList()
        for seed, fold_id in itertools.product(args.p_seeds, range(args.num_folds)):
            # ckpt_path = self.args.output_dir / f'run-{seed}' / f'fold-{fold_id}' / 'cls' / 'last.ckpt'
            ckpt_path = self.args.output_dir / f'run-{seed}' / f'fold-{fold_id}' / 'cls'
            for filepath in ckpt_path.iterdir():
                if filepath.name.startswith('best'):
                    ckpt_path = filepath
                    break
            else:
                raise RuntimeError

            self.models.append(MBModel.load_from_checkpoint(ckpt_path, strict=True, args=args))
            print(f'load model from {ckpt_path}')

        self.metrics: Mapping[str, torchmetrics.Metric] = nn.ModuleDict({
            k: metric_cls(num_classes=len(SUBGROUPS), average=average)
            for k, metric_cls, average in [
                ('auroc', torchmetrics.AUROC, AverageMethod.NONE),
                ('recall', torchmetrics.Recall, AverageMethod.NONE),
                ('precision', torchmetrics.Precision, AverageMethod.NONE),
                ('f1', torchmetrics.F1Score, AverageMethod.NONE),
                ('acc', torchmetrics.Accuracy, AverageMethod.MICRO),
            ]
        })
        self.case_outputs = []
        self.report = {}

    def on_test_epoch_start(self) -> None:
        for k, m in self.metrics.items():
            m.reset()
        self.case_outputs.clear()
        self.report = {}

    def test_step(self, batch: dict, *args, **kwargs):
        prob = None
        for model in self.models:
            logit = model.forward_cls(batch[DataKey.IMG], tta=self.args.do_tta)
            if prob is None:
                prob = logit.softmax(dim=-1)
            else:
                prob += logit.softmax(dim=-1)
        prob /= len(self.models)
        cls = batch[DataKey.CLS]
        for k, metric in self.metrics.items():
            metric(prob, cls)
        for i, case in enumerate(batch[MBDataKey.CASE]):
            self.case_outputs.append({
                'case': case,
                'sg': SUBGROUPS[cls[i].item()],
                **{
                    sg: prob[i, j].item()
                    for j, sg in enumerate(SUBGROUPS)
                },
                'prob': prob[i].cpu().numpy(),
            })

    def test_epoch_end(self, *args) -> None:
        self.report['n'] = len(self.case_outputs)
        for k, metric in self.metrics.items():
            m = metric.compute()
            if metric.average == AverageMethod.NONE:
                result = {}
                for i, sg in enumerate(SUBGROUPS):
                    result[sg] = m[i].item()
                result['macro avg'] = m.mean().item()
                self.report[k] = result
            else:
                self.report[k] = m.item()
        print(json.dumps(self.report, indent=4, ensure_ascii=False))

def main():
    parser = UMeIParser((MBTestArgs,), use_conf=True)
    args: MBTestArgs = parser.parse_args_into_dataclasses()[0]
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
        # limit_test_batches=1,
    )
    trainer.test(tester, dataloaders=datamodule.test_dataloader())
    results_df = pd.DataFrame.from_records(tester.case_outputs)
    results_df.to_csv(args.output_dir / 'test-results.csv', index=False)
    results_df.to_excel(args.output_dir / 'test-results.xlsx', index=False)
    with open(args.output_dir / 'report.json', 'w') as f:
        json.dump(tester.report, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
