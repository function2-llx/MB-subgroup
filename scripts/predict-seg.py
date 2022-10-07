from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn

import monai
from monai.data import DataLoader, Dataset
from monai.utils import GridSampleMode
from umei.utils import DataKey, UMeIParser

from mbs.args import MBSegArgs
from mbs.datamodule import MBSegDataModule
from mbs.model import MBSegModel
from mbs.utils.enums import MBDataKey

@dataclass
class MBSegPredictionArgs(MBSegArgs):
    p_seeds: list[int] = field(default=None)
    p_output_dir: Path = field(default=None)
    l: int = field(default=None)
    r: int = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.p_output_dir is None:
            suffix = f'sw{self.sw_overlap}'
            if self.do_post:
                suffix += '+post'
            if self.do_tta:
                suffix += '+tta'
            self.p_output_dir = self.output_dir / f'predict-{"+".join(map(str, self.p_seeds))}' / suffix

class MBSegPredictor(pl.LightningModule):
    models: nn.ModuleList | Sequence[MBSegModel]

    def __init__(self, args: MBSegPredictionArgs):
        super().__init__()
        self.args = args
        self.models = nn.ModuleList([
            MBSegModel.load_from_checkpoint(
                self.args.output_dir / f'run-{seed}' / f'fold-{fold_id}' / 'last.ckpt',
                strict=True,
                args=args,
            )
            for seed in args.p_seeds for fold_id in range(args.num_folds)
        ])

    def predict_step(self, batch: dict[str, ...], *args, **kwargs):
        case: str = batch[MBDataKey.CASE][0]
        img: torch.Tensor = batch[DataKey.IMG]
        pred_prob: Optional[torch.Tensor] = None
        for model in self.models:
            prob = model.infer_logit(img).sigmoid()
            if pred_prob is None:
                pred_prob = prob
            else:
                pred_prob += prob
        pred_prob /= len(self.models)
        save_path = self.args.p_output_dir / case / 'seg-prob.pt'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pred_prob, save_path)

class MBSegPredictionDataModule(MBSegDataModule):
    args: MBSegPredictionArgs

    @property
    def predict_transform(self):
        img_keys = self.args.input_modalities
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(img_keys),
            monai.transforms.EnsureChannelFirstD(img_keys),
            monai.transforms.OrientationD(img_keys, axcodes='RAS'),
            monai.transforms.SpacingD(img_keys, pixdim=self.args.spacing, mode=GridSampleMode.BILINEAR),
            monai.transforms.ResizeWithPadOrCropD(img_keys, spatial_size=self.args.pad_crop_size),
            monai.transforms.NormalizeIntensityD(img_keys),
            monai.transforms.LambdaD(img_keys, lambda t: t.as_tensor(), track_meta=False),
            monai.transforms.ConcatItemsD(img_keys, name=DataKey.IMG),
            monai.transforms.SelectItemsD([DataKey.IMG, MBDataKey.CASE]),
        ])

    def predict_dataloader(self):
        return DataLoader(
            dataset=Dataset(
                self.all_data()[self.args.l:self.args.r],
                transform=self.predict_transform,
            ),
            num_workers=self.args.dataloader_num_workers,
            batch_size=self.args.per_device_eval_batch_size,
            pin_memory=True,
            persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
        )

def main():
    parser = UMeIParser((MBSegPredictionArgs,), use_conf=True)
    args: MBSegPredictionArgs = parser.parse_args_into_dataclasses()[0]
    print(args)
    predictor = MBSegPredictor(args)
    trainer = pl.Trainer(
        logger=False,
        num_nodes=args.num_nodes,
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        precision=args.precision,
        benchmark=True,
    )
    datamodule = MBSegPredictionDataModule(args)
    trainer.predict(predictor, datamodule=datamodule)

if __name__ == '__main__':
    main()
