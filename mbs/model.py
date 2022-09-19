from __future__ import annotations

import itertools
from typing import Type

import einops
import torch
from torch import nn
from torch.nn import functional as torch_f
from torchmetrics import Recall

import monai
from monai.networks.blocks import Convolution, UnetBasicBlock, UnetResBlock
from monai.networks.layers import Act, Norm
from monai.networks.nets import PatchMergingV2
from monai.networks.nets.swin_unetr import BasicLayer
from monai.umei import UEncoderBase, UEncoderOutput
from umei import SegModel
from umei.models.layernorm import LayerNormNd
from umei.utils import DataKey

from mbs.args import MBSegArgs
from mbs.cnn_decoder import CNNDecoder
from mbs.utils.enums import MBDataKey, SegClass

class MBPatchMerging(PatchMergingV2):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
        *,
        z_stride: int
    ):
        super().__init__(dim, norm_layer, spatial_dims)
        self.out_dim = out_dim
        self.z_stride = z_stride
        self.reduction = nn.Linear(4 * z_stride * dim, out_dim, bias=False)
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
        args: MBSegArgs,
        drop_path_rate: float = 0.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.conv_stem = nn.ModuleList([
            UnetBasicBlock(
                spatial_dims=3,
                in_channels=args.num_input_channels,
                out_channels=args.feature_channels[0],
                kernel_size=3,
                stride=1,
                act_name=Act.GELU,
                norm_name=Norm.LAYERND,
            )
        ])
        for i in range(1, args.stem_stages):
            self.conv_stem.append(
                UnetResBlock(
                    spatial_dims=3,
                    in_channels=args.feature_channels[i - 1],
                    out_channels=args.feature_channels[i],
                    kernel_size=3,
                    stride=(2, 2, args.z_strides[i - 1]),
                    act_name=Act.GELU,
                    norm_name=Norm.LAYERND,
                )
            )
        self.patch_embed = Convolution(
            spatial_dims=3,
            in_channels=args.feature_channels[args.stem_stages - 1],
            out_channels=args.feature_channels[args.stem_stages],
            strides=(2, 2, args.z_strides[args.stem_stages - 1]),
            kernel_size=3,
            conv_only=True,
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
                downsample=MBPatchMerging(
                    dim=args.feature_channels[args.stem_stages + i_layer],
                    out_dim=args.feature_channels[args.stem_stages + i_layer + 1],
                    z_stride=args.z_strides[args.stem_stages + i_layer],
                ) if i_layer + 1 < args.vit_stages else None,
                use_checkpoint=args.gradient_checkpointing,
            )
            for i_layer in range(args.vit_stages)
        ])

        self.norms = nn.ModuleList([
            LayerNormNd(args.feature_channels[args.stem_stages + i])
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
            z = norm(z)
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
        self.recall = Recall(num_classes=args.num_seg_classes, average=None, multiclass=False)
        self.recall_u = Recall(num_classes=args.num_seg_classes, average=None, multiclass=False)

    def build_encoder(self):
        return MBBackbone(self.args)

    def build_decoder(self, encoder_feature_sizes: list[int]):
        return CNNDecoder(
            feature_channels=self.args.feature_channels,
            z_strides=self.args.z_strides,
            num_layers=self.args.num_stages,
        )

    def on_test_start(self) -> None:
        self.dice_metric.reset()
        self.recall.reset()
        self.recall_u.reset()
        self.test_outputs.clear()

    def test_step(self, batch, *args, **kwargs):
        seg = batch[DataKey.SEG].long()
        case = batch[MBDataKey.CASE][0]
        pred_logit = self.infer(batch[DataKey.IMG])

        pred = (pred_logit.sigmoid() > 0.5).long()
        if self.args.do_post:
            pred = self.post_transform(pred[0])[None]

        dice = self.dice_metric(pred, seg)
        recall = self.recall(
            einops.rearrange(pred, 'n c ... -> (n ...) c'),
            einops.rearrange(seg, 'n c ... -> (n ...) c'),
        )
        print(case, dice, recall)
        output = {
            MBDataKey.CASE: case,
            **{
                f'dice-{s}': dice[0, i].item()
                for i, s in enumerate(self.args.seg_classes)
            },
            **{
                f'recall-{s}': recall[i].item()
                for i, s in enumerate(self.args.seg_classes)
            }
        }
        if self.args.num_seg_classes == len(SegClass):
            # want reduce: https://github.com/pytorch/pytorch/issues/35641
            pred_u = pred[:, 0]
            for i in range(1, self.args.num_seg_classes):
                pred_u |= pred[:, i]
            pred_u = einops.repeat(pred_u, 'n ... -> n c ...', c=self.args.num_seg_classes)
            recall_u = self.recall_u(
                einops.rearrange(pred_u, 'n c ... -> (n ...) c'),
                einops.rearrange(seg, 'n c ... -> (n ...) c'),
            )
            print(recall_u)
            for i, s in enumerate(self.args.seg_classes):
                output[f'recall-u-{s}'] = recall_u[i].item()

        self.test_outputs.append(output)
