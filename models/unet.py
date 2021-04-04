# adopted from monai's implementation

from __future__ import annotations

from typing import Sequence, Union, Tuple

import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.utils import SkipMode

class UBlock(nn.Module):
    def __init__(self, down: nn.Module, subblock: nn.Module, up: nn.Module):
        super().__init__()
        self.down = down
        self.subblock = subblock
        self.up = up

    def forward(self, x):
        x = self.down(x)
        hidden, x = self.subblock(x)
        x = self.up(x)
        return hidden, x

class DoubleOutputLayer(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layer(x)
        return x, x

class DoubleSkipConnection(nn.Module):
    """
    Combine the forward pass input with the result from the given submodule::

        --+--submodule--o--
          |_____________|

    The available modes are ``"cat"``, ``"add"``, ``"mul"``.
    """

    def __init__(self, submodule, dim: int = 1, mode: Union[str, SkipMode] = "cat") -> None:
        """
        Args:
            submodule: the module defines the trainable branch.
            dim: the dimension over which the tensors are concatenated.
                Used when mode is ``"cat"``.
            mode: ``"cat"``, ``"add"``, ``"mul"``. defaults to ``"cat"``.
        """
        super().__init__()
        self.submodule = submodule
        self.dim = dim
        self.mode = SkipMode(mode).value

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, y = self.submodule(x)
        y = {
            'cat': lambda: torch.cat([x, y], dim=self.dim),
            'add': lambda: torch.add(x, y),
            'mul': lambda: torch.mul(x, y),
        }[self.mode]()
        return hidden, y

class UNet(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0.0,
        n_classes=None,
    ) -> None:
        """
        Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
        The residual part uses a convolution to change the input dimensions to match the output dimensions
        if this is necessary but will use nn.Identity if not.
        Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

        Args:
            dimensions: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            channels: sequence of channels. Top block first.
            strides: convolution stride.
            kernel_size: convolution kernel size. Defaults to 3.
            up_kernel_size: upsampling convolution kernel size. Defaults to 3.
            num_res_units: number of residual units. Defaults to 0.
            act: activation type and arguments. Defaults to PReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()

        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.n_classes = n_classes

        self.u_net = self._create_block(in_channels, out_channels, self.channels, self.strides, True)
        if n_classes is not None:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Linear(channels[-1], n_classes)

    def finetune_parameters(self, args):
        params = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        no_decay = ['bias', 'Norm.weight']
        grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay,
             'lr': args.lr},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.lr},
        ]
        return grouped_parameters

    def _create_block(
        self, inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
    ) -> UBlock:
        """
        Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
        blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

        Args:
            inc: number of input channels.
            outc: number of output channels.
            channels: sequence of channels. Top block first.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        c = channels[0]
        s = strides[0]

        subblock: nn.Module

        if len(channels) > 2:
            subblock = self._create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
            upc = c * 2
        else:
            # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
            subblock = self._get_bottom_layer(c, channels[1])
            upc = c + channels[1]

        down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
        up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

        return UBlock(down, DoubleSkipConnection(subblock), up)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
        return Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
        )

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        layer = self._get_down_layer(in_channels, out_channels, 1, False)
        return DoubleOutputLayer(layer)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_top,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor):
        hidden, seg = self.u_net(x)
        outputs = {'seg': seg}
        if self.n_classes is not None:
            hidden = self.avg_pool(hidden).view(hidden.shape[0], -1)
            outputs['linear'] = self.fc(hidden)

        return outputs
