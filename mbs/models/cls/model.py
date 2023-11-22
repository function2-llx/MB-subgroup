from collections.abc import Sequence

import einops
from einops.layers.torch import Reduce
import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.lightning import LightningModule
from luolib.models.blocks import sac
from luolib.models import BackboneProtocol, MaskFormer
from monai.inferers import sliding_window_inference
from monai.losses.focal_loss import sigmoid_focal_loss
from monai.metrics import DiceMetric
from monai.utils import MetricReduction

from .utils import DeepSupervisionWrapper, SlidingWindowInferenceConf

__all__ = [
    'MBSegModel',
    'MBSegMaskFormerModel',
    'MBSegUNetModel',
]

class MBSegModel(LightningModule):
    def __init__(
        self,
        *args,
        loss: nn.Module | None = None,
        sw: SlidingWindowInferenceConf,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss = loss
        self.sw_conf = sw
        self.dice_metric = DiceMetric()

    def patch_infer(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sw_infer(self, img: torch.Tensor, progress: bool = False, softmax: bool = False):
        conf = self.sw_conf
        ret = sliding_window_inference(
            img,
            conf.window_size,
            sw_batch_size=conf.batch_size,
            predictor=self.patch_infer,
            overlap=conf.overlap,
            mode=conf.blend_mode,
            progress=progress,
        )
        if softmax:
            ret = ret.softmax(dim=1)
        return ret

    @staticmethod
    def tta_flips(spatial_dims: int):
        match spatial_dims:
            case 2:
                return [[2], [3], [2, 3]]
            case 3:
                return [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
            case _:
                raise ValueError

    def tta_sw_infer(self, img: torch.Tensor, progress: bool = False, softmax: bool = False):
        pred = self.sw_infer(img, progress, softmax)
        spatial_dims = sum(s > 1 for s in img.shape[2:])
        tta_flips = self.tta_flips(spatial_dims)
        for flip_idx in tta_flips:
            pred += torch.flip(self.sw_infer(torch.flip(img, flip_idx), progress, softmax), flip_idx)
        pred /= len(tta_flips) + 1
        return pred

    def on_validation_epoch_start(self) -> None:
        self.dice_metric.reset()
        if self.sw_conf.check_window_size:
            from mbs.datamodule import MBSegDataModule
            dm: MBSegDataModule = self.datamodule
            assert self.sw_conf.window_size == dm.seg_trans_conf.patch_size

    def validation_step(self, batch: dict, *args, **kwargs):
        img, label = batch
        pred_logit = self.sw_infer(img)
        if pred_logit.shape[2:] != label.shape[2:]:
            pred_logit = sac.resample(pred_logit, label.shape[2:])
        loss = self.loss(pred_logit, label)
        self.log('val/loss', loss, sync_dist=True)
        pred = (pred_logit.sigmoid() > 0.5).long()
        self.dice_metric(pred, label)

    def on_validation_epoch_end(self) -> None:
        dice = self.dice_metric.aggregate(MetricReduction.MEAN_BATCH) * 100
        for i in range(dice.shape[0]):
            self.log(f'val/dice/{i}', dice[i])
        self.log(f'val/dice/avg', dice.mean())

class MBClsModel(MaskFormer, LightningModule):
    def __init__(
        self,
        *args,
        embed_dim: int,
        num_classes: int,
        num_cls_layers: int = 1,
        loss: nn.Module | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cls_head = nn.Sequential()
        for _ in range(num_cls_layers):
            self.cls_head.extend([nn.Linear(embed_dim, embed_dim), nn.ReLU()])
        self.cls_head.append(nn.Linear(embed_dim, num_classes))
        self.loss = loss

    def training_step(self, batch: dict, *args, **kwargs):
        img, mask, label = batch
        layers_mask_embeddings, _ = self(img)
        loss = layers_mask_embeddings[-1]
        self.log('train/loss', loss)
        return loss

    def patch_infer(self, img: torch.Tensor):
        layers_mask_embeddings, layers_mask_logits = self(img)
        logits = layers_mask_logits[-1][0]
        if logits.shape[2:] != img.shape[2:]:
            d = logits.shape[2]
            assert d == img.shape[2]
            logits_2d = nnf.interpolate(
                einops.rearrange(logits, 'n c d h w -> n (c d) h w'), img.shape[3:],
                mode='bicubic',
            )
            logits = einops.rearrange(logits_2d, 'n (c d) h w -> n c d h w', d=d)
        return logits
