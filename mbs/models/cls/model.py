from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import nn
import torchmetrics
from torchmetrics.utilities.enums import AverageMethod

from luolib.lightning import LightningModule
from luolib.models import MaskFormer
from luolib.utils import DataSplit

__all__ = [
    'MBClsModel',
]

from mbs.utils.enums import SUBGROUPS

class ClsMetricsCollection(nn.ModuleDict):
    def __init__(self, cls_names: list[str]):
        self.cls_names = list(cls_names)
        num_classes = len(cls_names)
        metrics = {
            metric_name: metric_cls(task='multiclass', num_classes=num_classes, average=average)
            for metric_name, metric_cls, average in [
                ('auroc', torchmetrics.AUROC, AverageMethod.NONE),
                ('recall', torchmetrics.Recall, AverageMethod.NONE),
                ('precision', torchmetrics.Precision, AverageMethod.NONE),
                ('f1', torchmetrics.F1Score, AverageMethod.NONE),
                ('acc', torchmetrics.Accuracy, AverageMethod.MICRO),
            ]
        }
        super().__init__(metrics)

    def items(self) -> Iterable[Tuple[str, torchmetrics.Metric]]:
        yield from super().items()

    def values(self) -> Iterable[torchmetrics.Metric]:
        yield from super().values()

    def reset(self):
        for metric in self.values():
            metric.reset()

    def accumulate(self, prob: torch.Tensor, pred: torch.Tensor, label: torch.Tensor):
        for name, metric in self.items():
            if name == 'auroc':
                metric(prob, label)
            else:
                metric(pred, label)

    def get_log_dict(self, prefix: str):
        ret = {}
        for name, metric in self.items():
            m = metric.compute()
            if metric.average == AverageMethod.NONE:
                for i, cls in enumerate(self.cls_names):
                    ret[f'{prefix}/{name}/{cls}'] = m[i]
                ret[f'{prefix}/{name}/avg'] = m.mean()
            else:
                ret[f'{prefix}/{name}'] = m
        return ret

class MBClsModel(MaskFormer, LightningModule):
    def __init__(
        self,
        *args,
        embed_dim: int,
        use_clinical: bool = False,
        num_cls_layers: int = 1,
        loss: nn.Module | None = None,
        pretrained_ckpt_path: Path | None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cls_head = nn.Sequential()
        last_dim = embed_dim
        if use_clinical:
            last_dim += 3
        for _ in range(num_cls_layers):
            self.cls_head.extend([nn.Linear(last_dim, embed_dim), nn.ReLU()])
            last_dim = embed_dim
        self.cls_head.append(nn.Linear(last_dim, len(SUBGROUPS)))
        self.use_clinical = use_clinical
        self.loss = loss
        self.metrics: Mapping[str, ClsMetricsCollection] = nn.ModuleDict({
            split: ClsMetricsCollection(SUBGROUPS)
            for split in [DataSplit.VAL, DataSplit.TEST]
        })
        self.pretrained_ckpt_path = pretrained_ckpt_path

    def on_fit_start(self) -> None:
        load_ckpt(self, self.pretrained_ckpt_path)
        super().on_fit_start()

    def patch_forward(self, img: torch.Tensor, mask: torch.Tensor):
        layers_mask_embeddings, _ = self(img, mask)
        logits = self.cls_head(layers_mask_embeddings[-1][:, 1])
        return logits

    def training_step(self, batch: dict, *args, **kwargs):
        img, mask, label = batch
        logits = self.patch_forward(img, mask)
        loss = self.loss(logits, label)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch: dict, _batch_idx, dataloader_idx: int):
        split = (DataSplit.VAL, DataSplit.TEST)[dataloader_idx]
        img, mask, label = batch
        logits = self.patch_forward(img, mask)
        loss = self.loss(logits, label)
        self.log(f'{split}/loss', loss)
        prob = logits.softmax(dim=-1)
        pred = prob.argmax(dim=-1)
        self.metrics[split].accumulate(prob, pred, label)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        for split in (DataSplit.VAL, DataSplit.TEST):
            self.log_dict(self.metrics[split].get_log_dict(split), sync_dist=True)
