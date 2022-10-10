from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import json

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import AUROC
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
        self.models = nn.ModuleList([
            MBModel.load_from_checkpoint(
                self.args.output_dir / f'run-{seed}' / f'fold-{fold_id}' / 'last.ckpt',
                strict=True,
                args=args,
            )
            for seed in args.p_seeds
            # for fold_id in range(args.num_folds)
            for fold_id in [4]
        ])
        self.auroc = AUROC(num_classes=args.num_cls_classes, average=AverageMethod.NONE)
        self.case_outputs = []
        self.whole_results = {}

    def on_test_epoch_start(self) -> None:
        self.auroc.reset()
        self.case_outputs.clear()

    def test_step(self, batch: dict, *args, **kwargs):
        prob = None
        for model in self.models:
            logit = model.forward_cls(batch[DataKey.IMG], tta=self.args.do_tta)
            if prob is None:
                prob = logit.softmax(dim=-1)
            else:
                prob += logit.softmax(dim=-1)
        prob /= len(self.models)
        self.auroc(prob, batch[DataKey.CLS])
        cls = batch[DataKey.CLS]
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
        auroc = self.auroc.compute()
        for i, sg in enumerate(SUBGROUPS):
            self.whole_results[sg] = auroc[i].item()
        self.whole_results['macro avg'] = auroc.mean().item()
        print(self.whole_results)

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
    with open(args.output_dir / 'whole-result.json', 'w') as f:
        json.dump(tester.whole_results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
