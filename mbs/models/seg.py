import torch

from luolib.lightning import LightningModule
from luolib.models.blocks import sac
from luolib.models import MaskFormer
from monai.losses import DiceFocalLoss

__all__ = [
    'MBSegModel',
]

class MBSegModel(MaskFormer, LightningModule):
    def __init__(self, *args, gamma: float = 0., **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_loss = DiceFocalLoss(sigmoid=True, gamma=gamma)

    def training_step(self, batch: dict, *args, **kwargs):
        layers_mask_embeddings, layers_mask_logits = self.decode_mask(batch['img'])
        label = batch['seg']
        pred_shape = layers_mask_logits[0].shape[2:]
        if pred_shape != label.shape[2:]:
            label = sac.resample(label, pred_shape)
        return torch.stack([self.mask_loss(mask_logit, label) for mask_logit in layers_mask_logits]).mean()
