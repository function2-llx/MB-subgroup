from dataclasses import dataclass

import torch

from luolib.lightning import LightningModule
from luolib.models.blocks import sac
from luolib.models import MaskFormer
from luolib.types import spatial_param_t
from luolib.utils import IndexTracker
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.utils import BlendMode, MetricReduction

__all__ = [
    'MBSegModel',
]


@dataclass
class SlidingWindowInferenceConf:
    window_size: spatial_param_t[int]
    batch_size: int = 4
    overlap: float = 0.5
    blend_mode: BlendMode = BlendMode.GAUSSIAN

class MBSegModel(MaskFormer, LightningModule):
    def __init__(
        self,
        *args,
        gamma: float = 0.,
        val_sw: SlidingWindowInferenceConf,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_loss = DiceFocalLoss(sigmoid=True, gamma=gamma)
        self.val_sw_conf = val_sw
        self.dice_metric = DiceMetric()

    def training_step(self, batch: dict, *args, **kwargs):
        img, label = batch
        for c in range(img.shape[1]):
            IndexTracker(img[0, c], label[0, 0], zyx=True)
        layers_mask_embeddings, layers_mask_logits = self(img)
        pred_shape = layers_mask_logits[0].shape[2:]
        if pred_shape != label.shape[2:]:
            label = sac.resample(label, pred_shape)
        loss = torch.stack([self.mask_loss(mask_logit, label) for mask_logit in layers_mask_logits]).mean()
        self.log('train/loss', loss)
        return loss

    def predict_seg(self, img: torch.Tensor):
        layers_mask_embeddings, layers_mask_logits = self(img)
        return layers_mask_logits[-1]

    def sw_infer(self, img: torch.Tensor, progress: bool = None, softmax: bool = False):
        conf = self.val_sw_conf
        ret = sliding_window_inference(
            img,
            conf.window_size,
            sw_batch_size=conf.batch_size,
            predictor=self.predict_seg,
            overlap=conf.overlap,
            mode=conf.blend_mode,
            progress=progress,
        )
        if softmax:
            ret = ret.softmax(dim=1)
        return ret

    def on_validation_epoch_start(self) -> None:
        self.dice_metric.reset()

    def validation_step(self, batch: dict, *args, **kwargs):
        img, label = batch
        pred_logit = self.sw_infer(img)
        if pred_logit.shape[2:] != label.shape[2:]:
            pred_logit = sac.resample(pred_logit, label.shape[2:])
        loss = self.mask_loss(pred_logit, label)
        self.log('val/loss', loss, sync_dist=True)
        pred = (pred_logit.sigmoid() > 0.5).long()
        self.dice_metric(pred, label)

    def on_validation_epoch_end(self) -> None:
        dice = self.dice_metric.aggregate(MetricReduction.MEAN_BATCH) * 100
        for i in range(dice.shape[0]):
            self.log(f'val/dice/{i}', dice[i])
        self.log(f'val/dice/avg', dice.mean())
