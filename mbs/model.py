from collections.abc import Mapping
import itertools
from pathlib import Path

from einops.layers.torch import Rearrange
import torch
from torch import nn
import torchmetrics
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from torchmetrics.utilities.enums import AverageMethod

from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.blocks import UnetResBlock
from monai.networks.layers import Pool

from luolib.utils import DataKey, DataSplit

# class MBBackbone(UEncoderBase):
#     def __init__(
#         self,
#         args: MBSegConf,
#         drop_path_rate: float = 0.0,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         drop_rate: float = 0.0,
#         attn_drop_rate: float = 0.0,
#     ):
#         super().__init__()
#         self.conv_stem = nn.ModuleList([
#             UnetBasicBlock(
#                 spatial_dims=3,
#                 in_channels=args.num_input_channels,
#                 out_channels=args.feature_channels[0],
#                 kernel_size=(3, 3, args.z_kernel_sizes[0]),
#                 stride=1,
#                 norm_name=args.conv_norm,
#                 act_name=args.conv_act,
#             )
#         ])
#         for i in range(1, args.stem_stages):
#             self.conv_stem.append(
#                 UnetResBlock(
#                     spatial_dims=3,
#                     in_channels=args.feature_channels[i - 1],
#                     out_channels=args.feature_channels[i],
#                     kernel_size=(3, 3, args.z_kernel_sizes[i]),
#                     stride=(2, 2, args.z_strides[i - 1]),
#                     norm_name=args.conv_norm,
#                     act_name=args.conv_act,
#                 )
#             )
#         self.downsamples = nn.ModuleList([
#             Convolution(
#                 spatial_dims=3,
#                 in_channels=args.feature_channels[i - 1],
#                 out_channels=args.feature_channels[i],
#                 strides=(2, 2, args.z_strides[i - 1]),
#                 kernel_size=(2, 2, args.z_strides[i - 1]),
#                 padding=0,
#                 bias=False,
#                 conv_only=True,
#             )
#             for i in range(args.stem_stages, args.num_stages)
#         ])
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(args.vit_depths))]
#         self.layers = nn.ModuleList([
#             BasicLayer(
#                 dim=args.feature_channels[args.stem_stages + i_layer],
#                 depth=args.vit_depths[i_layer],
#                 num_heads=args.vit_num_heads[i_layer],
#                 window_size=args.swin_window_size,
#                 drop_path=dpr[sum(args.vit_depths[:i_layer]): sum(args.vit_depths[:i_layer + 1])],
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 downsample=None,
#                 use_checkpoint=args.gradient_checkpointing,
#             )
#             for i_layer in range(args.vit_stages)
#         ])
#
#         self.norms = nn.ModuleList([
#             LayerNormNd(args.feature_channels[args.stem_stages + i])
#             for i in range(args.vit_stages)
#         ])
#         self.pool = None
#
#     def forward(self, x: torch.Tensor, *args, **kwargs) -> UEncoderOutput:
#         hidden_states = []
#         for conv in self.conv_stem:
#             x = conv(x)
#             hidden_states.append(x)
#         for layer, norm, downsample in zip(self.layers, self.norms, self.downsamples):
#             x = downsample(x)
#             # x = norm(layer(x))
#             x = layer(x)
#             hidden_states.append(x)
#         ret = UEncoderOutput(
#             cls_feature=hidden_states[-1],
#             hidden_states=hidden_states,
#         )
#         if self.pool is not None:
#             z = hidden_states[-1]
#             ret.cls_feature = self.pool(z).flatten(1)
#         return ret

