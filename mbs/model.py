from __future__ import annotations

from collections.abc import Mapping
import itertools
from typing import Type

import einops
import torch
from torch import nn
from torch.nn import functional as torch_f
import torchmetrics
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC
from torchmetrics.utilities.enums import AverageMethod

import monai
from monai.losses import FocalLoss
from monai.networks.blocks import Convolution, UnetBasicBlock, UnetResBlock
from monai.networks.layers import Act, Norm
from monai.networks.nets import PatchMergingV2
from monai.networks.nets.swin_unetr import BasicLayer
from monai.umei import UEncoderBase, UEncoderOutput
from umei import SegModel
from umei.models.layernorm import LayerNormNd
from umei.utils import DataKey

from mbs.args import MBArgs, MBSegArgs
from mbs.cnn_decoder import CNNDecoder
from mbs.utils.enums import MBDataKey, SUBGROUPS, SegClass

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
        if args.num_cls_classes is not None:
            self.cls_pool = nn.AdaptiveAvgPool3d(tuple(args.pool_shape))
        else:
            self.cls_pool = None

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
        ret = UEncoderOutput(
            cls_feature=hidden_states[-1],
            hidden_states=hidden_states,
        )
        if self.cls_pool is not None:
            ret.cls_feature = self.cls_pool(hidden_states[-1]).flatten(1)
        return ret

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

    def cal_metrics(self, pred: torch.Tensor, seg: torch.Tensor):
        dice = self.dice_metric(pred, seg)
        recall = self.recall(
            einops.rearrange(pred, 'n c ... -> (n ...) c'),
            einops.rearrange(seg, 'n c ... -> (n ...) c'),
        )
        output = {
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

        return output

    def test_step(self, batch, *args, **kwargs):
        seg = batch[DataKey.SEG].long()
        case = batch[MBDataKey.CASE][0]
        pred_logit = self.infer_logit(batch[DataKey.IMG])

        pred = (pred_logit.sigmoid() > 0.5).long()
        if self.args.do_post:
            pred = self.post_transform(pred[0])[None]
        metrics = self.cal_metrics(pred, seg)
        metrics[MBDataKey.CASE] = case

        self.test_outputs.append(metrics)

class MBModel(MBSegModel):
    args: MBArgs
    encoder: MBBackbone

    def __init__(self, args: MBArgs):
        super().__init__(args)
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(args.cls_weights))
        self.seg_loss_fn = FocalLoss(
            include_background=self.args.include_background,
            to_onehot_y=not args.mc_seg,
            weight=torch.tensor(args.seg_weights),
        )
        self.cls_metrics: Mapping[str, torchmetrics.Metric] = nn.ModuleDict({
            k: metric_cls(num_classes=len(SUBGROUPS), average=average)
            for k, metric_cls, average in [
                ('auroc', AUROC, AverageMethod.NONE),
                ('recall', Recall, AverageMethod.NONE),
                ('precision', Precision, AverageMethod.NONE),
                ('f1', F1Score, AverageMethod.NONE),
                ('acc', Accuracy, AverageMethod.MICRO),
            ]
        })

    def validation_step(self, batch: dict[str, torch.Tensor], *args, **kwargs):
        super().validation_step(batch, *args, **kwargs)
        cls = batch[DataKey.CLS]
        logit = self.forward_cls(batch[DataKey.IMG])
        loss = self.cls_loss_fn(logit, cls)
        self.log('val/cls_loss', loss, sync_dist=True)
        prob = logit.softmax(dim=-1)
        for k, metric in self.cls_metrics.items():
            metric(prob, cls)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        for k, metric in self.cls_metrics.items():
            metric.reset()

    def validation_epoch_end(self, *args) -> None:
        super().validation_epoch_end(*args)
        for k, metric in self.cls_metrics.items():
            m = metric.compute()
            if metric.average == AverageMethod.NONE:
                for i, subgroup in enumerate(SUBGROUPS):
                    self.log(f'val/{k}/{subgroup}', m[i], sync_dist=True)
                self.log(f'val/{k}/avg', m.mean(), sync_dist=True)
            else:
                self.log(f'val/{k}', m, sync_dist=True)
