from __future__ import annotations

import itertools
from typing import Type

from einops import rearrange
import torch
from torch import nn as nn
from torch.nn import LayerNorm, functional as torch_f
from torchmetrics import Recall

import monai
from monai.networks.blocks import Convolution, UnetBasicBlock
from monai.networks.layers import Act, Norm
from monai.networks.nets import PatchMergingV2
from monai.networks.nets.swin_unetr import BasicLayer
from monai.umei import UEncoderBase, UEncoderOutput
from umei import SegModel
from umei.utils import DataKey

from mbs.args import MBArgs, MBSegArgs
from mbs.cnn_decoder import CNNDecoder
from mbs.utils.enums import MBDataKey

class PatchMergingV3(PatchMergingV2):
    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3, *, z_stride: int):
        super().__init__(dim, norm_layer, spatial_dims)
        self.z_stride = z_stride
        self.reduction = nn.Linear(4 * z_stride * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * z_stride * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % self.z_stride == 1)
            if pad_input:
                x = torch_f.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % self.z_stride))
            x = torch.cat(
                [
                    x[:, i::2, j::2, k::self.z_stride, :]
                    for i, j, k in itertools.product(range(2), range(2), range(self.z_stride))
                ],
                dim=-1,
            )
        x = self.norm(x)
        x = self.reduction(x)
        return x

class MBBackbone(UEncoderBase):
    def __init__(
        self,
        args: MBArgs,
        drop_path_rate: float = 0.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        vit_norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
    ):
        super().__init__()
        self.conv_stem = nn.ModuleList([
            UnetBasicBlock(
                spatial_dims=3,
                in_channels=args.num_input_channels if i == 0 else args.feature_channels[i - 1],
                out_channels=args.feature_channels[i],
                kernel_size=3,
                stride=1 if i == 0 else (2, 2, args.z_strides[i - 1]),
                act_name=Act.PRELU,
                norm_name=Norm.INSTANCE,
            )
            for i in range(args.stem_stages)
        ])
        self.patch_embed = Convolution(
            spatial_dims=3,
            in_channels=args.feature_channels[args.stem_stages - 1],
            out_channels=args.feature_channels[args.stem_stages],
            strides=(2, 2, args.z_strides[args.stem_stages - 1]),
            kernel_size=3,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(args.vit_depths))]
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=args.feature_channels[args.stem_stages + i_layer],
                depth=args.vit_depths[i_layer],
                num_heads=args.vit_num_heads[i_layer],
                window_size=args.swin_window_size,
                drop_path=dpr[sum(args.vit_depths[:i_layer]): sum(args.vit_depths[:i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=vit_norm_layer,
                downsample=PatchMergingV3(
                    dim=args.feature_channels[args.stem_stages + i_layer],
                    norm_layer=vit_norm_layer,
                    z_stride=args.z_strides[args.stem_stages + i_layer],
                ) if i_layer + 1 < args.vit_stages else None,
                use_checkpoint=args.gradient_checkpointing,
            )
            for i_layer in range(args.vit_stages)
        ])

        self.norms = nn.ModuleList([
            vit_norm_layer(args.feature_channels[args.stem_stages + i])
            for i in range(args.vit_stages)
        ])
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> UEncoderOutput:
        hidden_states = []
        for conv in self.conv_stem:
            x = conv(x)
            hidden_states.append(x)
        x = self.patch_embed(x)
        for layer, norm in zip(self.layers, self.norms):
            z, z_ds = layer(x)
            z = rearrange(z, "n c d h w -> n d h w c")
            z = norm(z)
            z = rearrange(z, "n d h w c -> n c d h w")
            hidden_states.append(z)
            x = z_ds
        return UEncoderOutput(
            cls_feature=self.avg_pool(hidden_states[-1]).flatten(1),
            hidden_states=hidden_states,
        )

class MBSegModel(SegModel):
    args: MBSegArgs

    def __init__(self, args: MBSegArgs):
        super().__init__(args)
        self.post_transform = monai.transforms.Compose([
            monai.transforms.KeepLargestConnectedComponent(is_onehot=True),
        ])
        self.test_outputs = []
        self.recall = Recall(num_classes=1, multiclass=False)

    def build_encoder(self):
        return MBBackbone(self.args)

    def build_decoder(self, encoder_feature_sizes: list[int]):
        return CNNDecoder(
            # feature_size=self.args.base_feature_size,
            feature_channels=self.args.feature_channels,
            z_strides=self.args.z_strides,
            num_layers=self.args.num_stages,
        )

    def on_test_start(self) -> None:
        self.dice_metric.reset()
        self.recall.reset()
        self.test_outputs.clear()

    def test_step(self, batch, *args, **kwargs):
        seg = batch[DataKey.SEG].long()
        case = batch[MBDataKey.CASE][0]
        pred_logit = self.sw_infer(batch[DataKey.IMG])

        pred = (pred_logit.sigmoid() > 0.5).long()
        if self.args.use_post:
            pred = self.post_transform(pred[0])[None]
        dice = self.dice_metric(pred, seg).item()
        recall = self.recall(pred.view(-1), seg.view(-1)).item()
        print(case, dice, recall)
        self.test_outputs.append({
            MBDataKey.CASE: case,
            'dice': dice,
            'recall': recall,
        })
