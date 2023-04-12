from collections.abc import Sequence

import torch
from torch import nn

from monai.networks.blocks import Convolution, UnetUpBlock
from monai.luolib import Decoder, DecoderOutput

class CNNDecoder(Decoder):
    def __init__(
        self,
        feature_channels: list[int],
        z_kernel_sizes: list[int] = None,
        z_strides: list[int] = None,
        num_layers: int = 4,
        norm_name: tuple | str = 'instance',
        act_name: tuple | str = 'leakyrelu',
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        assert spatial_dims == 3

        self.ups: Sequence[UnetUpBlock] | nn.ModuleList = nn.ModuleList([
            UnetUpBlock(
                spatial_dims,
                in_channels=feature_channels[i],
                out_channels=feature_channels[i - 1],
                kernel_size=(3, 3, z_kernel_sizes[i]),
                stride=-1,
                upsample_kernel_size=(2, 2, z_strides[i - 1]),
                norm_name=norm_name,
                act_name=act_name,
            )
            for i in range(1, num_layers)
        ])

        self.lateral_convs = nn.ModuleList([
            Convolution(
                spatial_dims,
                in_channels=feature_channels[i],
                out_channels=feature_channels[i],
                kernel_size=1,
                strides=1,
                bias=False,
                norm=norm_name,
                act=act_name,
            )
            for i in range(num_layers - 1)
        ])

    def forward(self, hidden_states: list[torch.Tensor], x_in: torch.Tensor) -> UDecoderOutput:
        x = hidden_states[-1]
        feature_maps = []
        for z, lateral_conv, up in zip(hidden_states[-2::-1], self.lateral_convs[::-1], self.ups[::-1]):
            z = lateral_conv(z)
            x = up.forward(x, z)
            feature_maps.append(x)
        return UDecoderOutput(feature_maps)
