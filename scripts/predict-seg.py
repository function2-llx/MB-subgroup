from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn

from mbs.utils import SEG_PROB_FILENAME
import monai
from monai.data import DataLoader, Dataset
from monai.utils import GridSampleMode
from umei.utils import DataKey, UMeIParser

from mbs.args import MBSegPredArgs
from mbs.datamodule import MBSegDataModule
from mbs.model import MBSegModel
from mbs.utils.enums import MBDataKey

class MBSegPredictor(pl.LightningModule):
    models: nn.ModuleList | Sequence[MBSegModel]

    def __init__(self, args: MBSegPredArgs):
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
        save_path = self.args.p_output_dir / case / SEG_PROB_FILENAME
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pred_prob, save_path)

class MBSegPredictionDataModule(MBSegDataModule):
    args: MBSegPredArgs

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
    parser = UMeIParser((MBSegPredArgs,), use_conf=True)
    args: MBSegPredArgs = parser.parse_args_into_dataclasses()[0]
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
