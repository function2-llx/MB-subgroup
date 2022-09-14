from __future__ import annotations

import torch
from torch import nn

from monai.networks.blocks import UnetResBlock, UnetrUpBlock
from monai.umei import UDecoderBase, UDecoderOutput

class CNNDecoder(UDecoderBase):
    def __init__(
        self,
        z_strides: list[int] = None,
        feature_size: int = 24,
        num_layers: int = 4,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        assert spatial_dims == 3

        self.bottleneck = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size << num_layers - 1,
            out_channels=feature_size << num_layers - 1,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.ups = nn.ModuleList([
            UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size << i,
                out_channels=feature_size << i - 1,
                kernel_size=3,
                upsample_kernel_size=(2, 2, z_strides[i - 1]),
                norm_name=norm_name,
                res_block=True,
            )
            for i in range(1, num_layers)
        ])

        self.lateral_convs = nn.ModuleList([
            UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size << i,
                out_channels=feature_size << i,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
            )
            for i in range(num_layers - 1)
        ])

    def forward(self, hidden_states: list[torch.Tensor], x_in: torch.Tensor) -> UDecoderOutput:
        x = self.bottleneck(hidden_states[-1])
        feature_maps = []
        for z, lateral_conv, up in zip(hidden_states[-2::-1], self.lateral_convs[::-1], self.ups[::-1]):
            up: UnetrUpBlock
            z = lateral_conv(z)
            x = up.forward(x, z if up.use_skip else None)
            feature_maps.append(x)
        return UDecoderOutput(feature_maps)
