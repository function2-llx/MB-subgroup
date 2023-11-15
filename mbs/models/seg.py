from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import einops
from einops.layers.torch import Reduce
import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.lightning import LightningModule
from luolib.models.blocks import sac
from luolib.models import BackboneProtocol, MaskFormer
from luolib.types import spatial_param_t
from monai.inferers import sliding_window_inference
from monai.losses.focal_loss import sigmoid_focal_loss
from monai.metrics import DiceMetric
from monai.utils import BlendMode, MetricReduction

__all__ = [
    'MBSegModel',
    'MBSegMaskFormerModel',
    'MBSegUNetModel',
]

@dataclass
class SlidingWindowInferenceConf:
    window_size: spatial_param_t[int]
    batch_size: int = 4
    overlap: float = 0.5
    blend_mode: BlendMode = BlendMode.GAUSSIAN
    check_window_size: bool = True

class MBSegModel(LightningModule):
    def __init__(
        self,
        *args,
        loss: nn.Module,
        val_sw: SlidingWindowInferenceConf,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss = loss
        self.val_sw_conf = val_sw
        self.dice_metric = DiceMetric()

    def predictor(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sw_infer(self, img: torch.Tensor, progress: bool = None, softmax: bool = False):
        conf = self.val_sw_conf
        ret = sliding_window_inference(
            img,
            conf.window_size,
            sw_batch_size=conf.batch_size,
            predictor=self.predictor,
            overlap=conf.overlap,
            mode=conf.blend_mode,
            progress=progress,
        )
        if softmax:
            ret = ret.softmax(dim=1)
        return ret

    def on_validation_epoch_start(self) -> None:
        self.dice_metric.reset()
        if self.val_sw_conf.check_window_size:
            from mbs.datamodule import MBSegDataModule
            dm: MBSegDataModule = self.datamodule
            assert self.val_sw_conf.window_size == dm.seg_trans_conf.patch_size

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

def binary_kl_div(logit_p: torch.Tensor, logit_q: torch.Tensor):
    """
    calculate binary KL divergence p * ln p/q + (1-p) * ln (1-p)/(1-q)
    """
    log_p = nnf.logsigmoid(logit_p)
    log_q = nnf.logsigmoid(logit_q)
    p = logit_p.sigmoid()
    # log_p - logit_p = log(1 - p)
    return p * (log_p - log_q) + (1 - p) * (log_p - logit_p - log_q + logit_q)

def symmetric_binary_kl_div(logit_p: torch.Tensor, logit_q: torch.Tensor):
    return binary_kl_div(logit_p, logit_q) + binary_kl_div(logit_q, logit_p)

class AdjacentLayerRegLoss(nn.Module):
    def __init__(
        self,
        *,
        hard: bool = True,
        smooth_nr: float = 1e-6,
        smooth_dr: float = 1e-6,
        dice_weight: float = 0.,
        focal_weight: float = 0.,
        focal_gamma: float = 0.,
    ):
        super().__init__()
        self.hard = hard
        self.spatial_sum = Reduce('n c ... -> n c', 'sum')
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma

    def dice(self, prob: torch.Tensor, target: torch.Tensor):
        intersection = self.spatial_sum(prob * target)
        denominator = self.spatial_sum(prob) + self.spatial_sum(target)
        dice = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        return dice.mean()

    def forward(self, last_logits: torch.Tensor, logits: torch.Tensor):
        with torch.no_grad():
            target = logits.sigmoid()
            if self.hard:
                target = target > 0.5
        last_prob = last_logits.sigmoid()
        dice = self.dice(last_prob, target)
        focal = sigmoid_focal_loss(last_logits, target, self.focal_gamma).mean()
        return self.dice_weight * dice + self.focal_weight * focal, dice, focal

class DeepSupervisionWrapper(nn.Module):
    weight: torch.Tensor

    def __init__(self, loss: nn.Module, num_ds: int):
        super().__init__()
        self.loss = loss
        weight = torch.tensor([1 / (1 << i) for i in range(num_ds)])
        weight /= weight.sum()
        self.register_buffer('weight', weight)

    @staticmethod
    @lru_cache(1)
    def prepare_labels(label: torch.Tensor, spatial_shapes: list[torch.Size]) -> list[torch.Tensor]:
        label_shape = label.shape[2:]
        return [
            nnf.interpolate(label, shape, mode='nearest-exact') if label_shape != shape else label
            for shape in spatial_shapes
        ]

    def forward(self, deep_logits: list[torch.Tensor], label: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        spatial_shapes = [logits.shape[2:] for logits in deep_logits]
        deep_labels = self.prepare_labels(label, spatial_shapes)
        ds_losses = [
            self.loss(logits, deep_label)
            for logits, deep_label in zip(deep_logits, deep_labels)
        ]
        ds_loss = torch.dot(self.weight, torch.stack(ds_losses))
        return ds_loss, ds_losses

class MBSegMaskFormerModel(MaskFormer, MBSegModel):
    def __init__(self, *, adjacent_layer_reg: AdjacentLayerRegLoss | None = None, **kwargs):
        """
        Args:
            alr: adjacent layer regularization
        """
        super().__init__(**kwargs)
        if adjacent_layer_reg is None:
            adjacent_layer_reg = AdjacentLayerRegLoss()
        self.adjacent_layer_reg = adjacent_layer_reg
        self.ds_wrapper = DeepSupervisionWrapper(self.loss, len(self.pixel_embedding_levels))

    def training_step(self, batch: dict, *args, **kwargs):
        img, label = batch
        layers_mask_embeddings, layers_mask_logits = self(img)
        labels = self.ds_wrapper.prepare_labels()
        mask_loss = torch.stack([
            self.ds_wrapper(mask_logits, label)[0] for mask_logits in layers_mask_logits
        ]).mean()
        layers_reg_losses = []
        for i in range(1, len(layers_mask_logits)):
            layer_reg_loss, layer_reg_dice, layer_reg_focal = self.adjacent_layer_reg(layers_mask_logits[i - 1][0], layers_mask_logits[i][0])
            layers_reg_losses.append(layer_reg_loss)
            self.log(f'train/reg-dice/layer-{i}', layer_reg_dice)
            self.log(f'train/reg-focal/layer-{i}', layer_reg_focal)
        reg_loss = torch.stack(layers_reg_losses).mean()
        self.log('train/reg_loss', reg_loss)
        loss = mask_loss + reg_loss
        self.log('train/loss', loss)
        return loss

    def predictor(self, img: torch.Tensor):
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

class MBSegUNetModel(MBSegModel):
    ds_weights: torch.Tensor

    def __init__(
        self,
        *,
        backbone: nn.Module,
        num_channels: Sequence[int],
        num_classes: int,
        num_ds: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(backbone, BackboneProtocol)
        self.backbone = backbone
        self.num_classes = num_classes
        num_channels = num_channels[:num_ds]
        self.seg_heads = nn.ModuleList([
            nn.Conv3d(c, num_classes, 1)
            for c in num_channels
        ])
        self.ds_wrapper = DeepSupervisionWrapper(self.loss, num_ds)

    def forward(self, x: torch.Tensor, ds: bool = False):
        feature_maps = self.backbone(x)
        if ds:
            return [
                seg_head(feature_map)
                for seg_head, feature_map in zip(self.seg_heads, feature_maps)
            ]
        else:
            return self.seg_heads[0](feature_maps[0])

    def training_step(self, batch: dict, *args, **kwargs):
        img, label = batch
        mask_logits = self(img, ds=True)
        ds_loss, ds_losses = self.ds_wrapper(mask_logits, label)
        self.log('train/single_loss', ds_losses[0])
        self.log('train/ds_loss', ds_loss)
        return ds_loss

    def predictor(self, img: torch.Tensor):
        return self(img)
