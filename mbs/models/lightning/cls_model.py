from luolib.models import ExpModelBase
from mbs.conf import MBClsConf

class MBClsModel(ExpModelBase):
    conf: MBClsConf

    def __init__(self, args: MBClsConf):
        super().__init__(args)
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(args.cls_weights))
        self.seg_loss_fn = DiceFocalLoss(
            include_background=self.conf.include_background,
            to_onehot_y=not args.mc_seg,
            sigmoid=args.mc_seg,
            softmax=not args.mc_seg,
            squared_pred=self.conf.squared_dice,
            smooth_nr=self.conf.dice_nr,
            smooth_dr=self.conf.dice_dr,
            focal_weight=args.seg_weights,
        )
        self.val_keys = [DataSplit.VAL, DataSplit.TEST]
        self.cls_metrics: Mapping[str, Mapping[str, torchmetrics.Metric]] = nn.ModuleDict({
            split: nn.ModuleDict({
                k: metric_cls(task='multiclass', num_classes=args.num_cls_classes, average=average)
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

        if args.cls_conv:
            modules = [
                UnetResBlock(
                    spatial_dims=3,
                    in_channels=args.feature_channels[-1],
                    out_channels=args.cls_feature_size,
                    kernel_size=3,
                    stride=2,
                    norm_name=args.conv_norm,
                ),
            ]
            if self.conf.addi_conv:
                modules.extend([
                    nn.Conv3d(cls_feature_size, cls_feature_size, (3, 3, 2)),
                    # LayerNormNd(args.cls_hidden_size),
                    nn.LeakyReLU(),
                ])

            modules.extend([
                Pool[args.pool_name, 3](1),
                Rearrange('n c 1 1 1 -> n c'),
                nn.Linear(args.cls_feature_size, args.num_cls_classes),
            ])

            self.cls_head = nn.Sequential(*modules)
        else:
            if args.cls_hidden_size is None:
                self.cls_head = nn.Sequential(
                    Pool[args.pool_name, 3](1),
                    Rearrange('n c 1 1 1 -> n c'),
                    nn.Linear(args.cls_feature_size, args.num_cls_classes),
                )
            else:
                self.cls_head = nn.Sequential(
                    Pool[args.pool_name, 3](1),
                    Rearrange('n c 1 1 1 -> n c'),
                    nn.Linear(args.feature_channels[-1], args.cls_feature_size),
                    nn.PReLU(args.cls_feature_size),
                    nn.Linear(args.cls_feature_size, args.num_cls_classes),
                )

    def load_seg_state_dict(self, seg_ckpt_path: Path):
        seg_state_dict = torch.load(seg_ckpt_path)['state_dict']
        input_weight_key = 'encoder.conv_stem.0.conv1.conv.weight'
        input_weight = seg_state_dict[input_weight_key]
        shape = input_weight.shape
        new_input_weight = torch.zeros(shape[0], self.conf.num_input_channels, *shape[2:], dtype=input_weight.dtype)
        new_input_weight[:, :len(self.conf.input_modalities)] = input_weight
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
                    for i, cls in enumerate(self.conf.cls_names):
                        self.log(f'{split}/{k}/{cls}', m[i], sync_dist=True)
                    self.log(f'{split}/{k}/avg', m.mean(), sync_dist=True)
                else:
                    self.log(f'{split}/{k}', m, sync_dist=True)

    def get_lr_scheduler(self, optimizer):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        return ReduceLROnPlateau(
            optimizer,
            mode=self.conf.monitor_mode,
            factor=self.conf.lr_reduce_factor,
            patience=self.conf.patience,
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
                'lr': self.conf.finetune_lr,
            },
            {
                'params': self.cls_head.parameters(),
                'lr': self.conf.learning_rate,
            }
        ]
