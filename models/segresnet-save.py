from dataclasses import dataclass

import torch
import torch.nn as nn
from monai.networks.nets import SegResNetVAE as SegResNetVAEBase, ResNet

from .backbone import Backbone

@dataclass
class SegResNetOutput:
    cls: torch.FloatTensor
    # cls_loss: torch.FloatTensor
    seg: torch.Tensor
    # seg_loss: torch.FloatTensor
    vae_loss: torch.FloatTensor = None

class SegResNetVAE(SegResNetVAEBase, Backbone):
    def __init__(
        self,
        num_classes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        if num_classes:
            self.fc = nn.Linear(49 * self.init_filters * 2 ** (len(self.down_layers) - 1), num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d((7, 7, 1))

    def forward(self, x: torch.FloatTensor) -> SegResNetOutput:
        net_input = x

        x, down_x = self.encode(x)
        logits = self.fc(self.avgpool(x).view(x.shape[0], -1))

        vae_input = x

        down_x.reverse()
        x = self.decode(x, down_x)

        vae_loss = None
        if self.training:
            vae_loss = self._get_vae_loss(net_input, vae_input)
        return SegResNetOutput(cls=logits, seg=x, vae_loss=vae_loss)
