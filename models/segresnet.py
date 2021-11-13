from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from monai.networks.nets import SegResNetVAE as SegResNetVAEBase
from monai.losses import DiceLoss

@dataclass
class SegResNetOutput:
    cls: torch.FloatTensor
    # cls_loss: torch.FloatTensor
    seg: torch.Tensor
    # seg_loss: torch.FloatTensor
    vae_loss: torch.FloatTensor = None

class SegResNetVAE(SegResNetVAEBase):
    def __init__(
        self,
        num_classes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        if num_classes:
            self.fc = nn.Linear(self.init_filters * 2 ** len(self.down_layers), num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.FloatTensor) -> SegResNetOutput:
        net_input = x

        x, down_x = self.encode(x)
        logits = self.fc(self.avgpool(x))

        vae_input = x

        down_x.reverse()
        x = self.decode(x, down_x)

        vae_loss = None
        if self.training:
            vae_loss = self._get_vae_loss(net_input, vae_input)
        return SegResNetOutput(cls=logits, seg=x, vae_loss=vae_loss)
