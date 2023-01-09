from collections.abc import Mapping
import itertools
from pathlib import Path

import einops
from einops.layers.torch import Rearrange
import torch
from torch import nn
import torchmetrics
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from torchmetrics.utilities.enums import AverageMethod

import monai
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.blocks import Convolution, UnetBasicBlock, UnetResBlock
from monai.networks.layers import Pool
from monai.networks.nets.swin_unetr import BasicLayer
from monai.umei import UEncoderBase, UEncoderOutput
from umei import SegModel
from umei.models.layernorm import LayerNormNd
from umei.utils import DataKey, DataSplit

from mbs.args import MBArgs, MBSegArgs
from mbs.cnn_decoder import CNNDecoder
from mbs.utils.enums import MBDataKey, SegClass

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
                kernel_size=(3, 3, args.z_kernel_sizes[0]),
                stride=1,
                norm_name=args.conv_norm,
                act_name=args.conv_act,
            )
        ])
        for i in range(1, args.stem_stages):
            self.conv_stem.append(
                UnetResBlock(
                    spatial_dims=3,
                    in_channels=args.feature_channels[i - 1],
                    out_channels=args.feature_channels[i],
                    kernel_size=(3, 3, args.z_kernel_sizes[i]),
                    stride=(2, 2, args.z_strides[i - 1]),
                    norm_name=args.conv_norm,
                    act_name=args.conv_act,
                )
            )
        self.downsamples = nn.ModuleList([
            Convolution(
                spatial_dims=3,
                in_channels=args.feature_channels[i - 1],
                out_channels=args.feature_channels[i],
                strides=(2, 2, args.z_strides[i - 1]),
                kernel_size=(2, 2, args.z_strides[i - 1]),
                padding=0,
                bias=False,
                conv_only=True,
            )
            for i in range(args.stem_stages, args.num_stages)
        ])
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
                downsample=None,
                use_checkpoint=args.gradient_checkpointing,
            )
            for i_layer in range(args.vit_stages)
        ])

        self.norms = nn.ModuleList([
            LayerNormNd(args.feature_channels[args.stem_stages + i])
            for i in range(args.vit_stages)
        ])
        self.pool = None

    def forward(self, x: torch.Tensor, *args, **kwargs) -> UEncoderOutput:
        hidden_states = []
        for conv in self.conv_stem:
            x = conv(x)
            hidden_states.append(x)
        for layer, norm, downsample in zip(self.layers, self.norms, self.downsamples):
            x = downsample(x)
            x = norm(layer(x))
            hidden_states.append(x)
        ret = UEncoderOutput(
            cls_feature=hidden_states[-1],
            hidden_states=hidden_states,
        )
        if self.pool is not None:
            z = hidden_states[-1]
            ret.cls_feature = self.pool(z).flatten(1)
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

    def build_decoder(self):
        return CNNDecoder(
            feature_channels=self.args.feature_channels,
            z_kernel_sizes=self.args.z_kernel_sizes,
            z_strides=self.args.z_strides,
            num_layers=self.args.num_stages,
            norm_name=self.args.conv_norm,
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
        self.seg_loss_fn = DiceFocalLoss(
            include_background=self.args.include_background,
            to_onehot_y=not args.mc_seg,
            sigmoid=args.mc_seg,
            softmax=not args.mc_seg,
            squared_pred=self.args.squared_dice,
            smooth_nr=self.args.dice_nr,
            smooth_dr=self.args.dice_dr,
            focal_weight=args.seg_weights,
        )
        self.val_keys = [DataSplit.VAL, DataSplit.TEST]
        self.cls_metrics: Mapping[str, Mapping[str, torchmetrics.Metric]] = nn.ModuleDict({
            split: nn.ModuleDict({
                k: metric_cls(num_classes=args.num_cls_classes, average=average)
                for k, metric_cls, average in [
                    ('auroc', AUROC, AverageMethod.NONE),
                    ('recall', Recall, AverageMethod.NONE),
                    ('precision', Precision, AverageMethod.NONE),
                    ('f1', F1Score, AverageMethod.NONE),
                    ('acc', Accuracy, AverageMethod.MICRO),
                ]
            })
            for split in self.val_keys
        })
        self.dice_metrics = {
            split: DiceMetric(include_background=True)
            for split in self.val_keys
        }

        cls_feature_size = args.cls_hidden_size + args.clinical_feature_size
        if args.cls_conv:
            modules = [
                UnetResBlock(
                    spatial_dims=3,
                    in_channels=args.feature_channels[-1],
                    out_channels=cls_feature_size,
                    kernel_size=3,
                    stride=2,
                    norm_name=args.conv_norm,
                ),
            ]
            if self.args.addi_conv:
                modules.extend([
                    nn.Conv3d(cls_feature_size, cls_feature_size, (3, 3, 2)),
                    # LayerNormNd(args.cls_hidden_size),
                    nn.LeakyReLU(),
                ])

            modules.extend([
                Pool[args.pool_name, 3](1),
                Rearrange('n c 1 1 1 -> n c'),
                nn.Linear(cls_feature_size, args.num_cls_classes),
            ])

            self.cls_head = nn.Sequential(*modules)
        else:
            if args.cls_hidden_size is None:
                self.cls_head = nn.Sequential(
                    Pool[args.pool_name, 3](1),
                    Rearrange('n c 1 1 1 -> n c'),
                    nn.Linear(args.feature_channels[-1] + args.clinical_feature_size, args.num_cls_classes),
                )
            else:
                self.cls_head = nn.Sequential(
                    Pool[args.pool_name, 3](1),
                    Rearrange('n c 1 1 1 -> n c'),
                    nn.Linear(args.feature_channels[-1], cls_feature_size),
                    nn.PReLU(cls_feature_size),
                    nn.Linear(cls_feature_size, args.num_cls_classes),
                )

    def load_seg_state_dict(self, seg_ckpt_path: Path):
        seg_state_dict = torch.load(seg_ckpt_path)['state_dict']
        input_weight_key = 'encoder.conv_stem.0.conv1.conv.weight'
        input_weight = seg_state_dict[input_weight_key]
        shape = input_weight.shape
        new_input_weight = torch.zeros(shape[0], self.args.num_input_channels, *shape[2:], dtype=input_weight.dtype)
        new_input_weight[:, :len(self.args.input_modalities)] = input_weight
        seg_state_dict[input_weight_key] = new_input_weight
        missing_keys, unexpected_keys = self.load_state_dict(
            seg_state_dict,
            strict=False,
        )
        assert len(unexpected_keys) == 0
        print(missing_keys)
        for k in missing_keys:
            assert k.startswith('cls_head') or k.startswith('cls_loss_fn')
        print(f'[INFO] load seg model weights from {seg_ckpt_path}')

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx, dl_idx: int):
        split = self.val_keys[dl_idx]
        # adapt implementation of super class
        self.dice_metric = self.dice_metrics[split]
        super().validation_step(batch, batch_idx, dl_idx)
        cls = batch[DataKey.CLS]
        logit = self.forward_cls(batch[DataKey.IMG])

        loss = self.cls_loss_fn(logit, cls)
        self.log(f'{split}/cls_loss', loss, sync_dist=True, add_dataloader_idx=False)
        prob = logit.softmax(dim=-1)
        for k, metric in self.cls_metrics[split].items():
            metric(prob, cls)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        for split in self.val_keys:
            for k, metric in self.cls_metrics[split].items():
                metric.reset()

    def validation_epoch_end(self, *args) -> None:
        super().validation_epoch_end(*args)
        for split in self.val_keys:
            for k, metric in self.cls_metrics[split].items():
                m = metric.compute()
                if metric.average == AverageMethod.NONE:
                    for i, cls in enumerate(self.args.cls_names):
                        self.log(f'{split}/{k}/{cls}', m[i], sync_dist=True)
                    self.log(f'{split}/{k}/avg', m.mean(), sync_dist=True)
                else:
                    self.log(f'{split}/{k}', m, sync_dist=True)

    def get_lr_scheduler(self, optimizer):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        return ReduceLROnPlateau(
            optimizer,
            mode=self.args.monitor_mode,
            factor=self.args.lr_reduce_factor,
            patience=self.args.patience,
            verbose=True,
        )

    def get_grouped_parameters(self) -> list[dict]:
        return [
            {
                'params': itertools.chain(
                    self.encoder.parameters(),
                    self.decoder.parameters(),
                    self.seg_heads.parameters(),
                ),
                'lr': self.args.finetune_lr,
            },
            {
                'params': self.cls_head.parameters(),
                'lr': self.args.learning_rate,
            }
        ]
