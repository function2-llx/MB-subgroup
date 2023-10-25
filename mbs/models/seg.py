from monai.losses import DiceFocalLoss
from luolib.models import MaskFormer

class MBSegModel(MaskFormer):
    def __init__(self, gamma: float = 0., **kwargs):
        super().__init__(**kwargs)
        self.mask_loss = DiceFocalLoss(sigmoid=True, gamma=gamma)

    def training_step(self, batch: dict, *args, **kwargs):
        layers_mask_embeddings, layers_mask_logits = self.decode_mask(batch['img'])
