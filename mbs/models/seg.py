from collections.abc import Sequence
from dataclasses import dataclass

from einops.layers.torch import Reduce
import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.lightning import LightningModule
from luolib.models.blocks import sac
from luolib.models import BackboneProtocol, MaskFormer
from luolib.types import spatial_param_t
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
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

class MBSegModel(LightningModule):
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
        dice_weight: float = 0.1,
        focal_weight: float = 0.1,
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
            target = last_logits.sigmoid()
            if self.hard:
                target = target > 0.5
        prob = logits.sigmoid()
        dice = self.dice(prob, target)
        focal = sigmoid_focal_loss(logits, target, self.focal_gamma).mean()
        return self.dice_weight * dice + self.focal_weight * focal, dice, focal

class MBSegMaskFormerModel(MaskFormer, MBSegModel):
    def __init__(self, *, adjacent_layer_reg: AdjacentLayerRegLoss, **kwargs):
        """
        Args:
            alr: adjacent layer regularization
        """
        super().__init__(**kwargs)
        self.adjacent_layer_reg = adjacent_layer_reg

    def training_step(self, batch: dict, *args, **kwargs):
        img, label = batch
        layers_mask_embeddings, layers_mask_logits = self(img)
        pred_shape = layers_mask_logits[0].shape[2:]
        if pred_shape != label.shape[2:]:
            label = sac.resample(label, pred_shape)
        label = (label.float() > 0.5).type_as(layers_mask_logits[0])
        mask_loss = torch.stack([self.mask_loss(mask_logit, label) for mask_logit in layers_mask_logits]).mean()
        self.log('train/mask_loss', mask_loss)
        layers_reg_losses = []
        for i in range(1, len(layers_mask_logits)):
            layer_reg_loss, layer_reg_dice, layer_reg_focal = self.adjacent_layer_reg(layers_mask_logits[i - 1], layers_mask_logits[i])
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
        return layers_mask_logits[-1]

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
        ds_weights = torch.tensor([0.5 ** i for i in range(num_ds)])
        ds_weights /= ds_weights.sum()
        self.register_buffer('ds_weights', ds_weights, persistent=False)

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
        ds_losses = torch.stack([
            self.mask_loss(mask_logit, (sac.resample(label, mask_logit.shape[2:]) >= 0.5).type_as(mask_logit))
            for mask_logit in mask_logits
        ])
        ds_loss = torch.dot(self.ds_weights, ds_losses)
        self.log('train/single_loss', ds_losses[0])
        self.log('train/ds_loss', ds_loss)
        return ds_loss

    def predictor(self, img: torch.Tensor):
        return self(img)
