import cytoolz
import einops
from lightning_fabric.utilities import move_data_to_device
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from tqdm import tqdm

from luolib.models import ClsModel, Mask2Former
from luolib.models.init import init_common
from luolib.models.utils import get_no_weight_decay_keys
# from luolib.types import NamedParamGroup
from luolib.utils import DataKey

from mbs.conf import MBClsConf, get_cls_names, MBM2FClsConf
from mbs.utils.enums import SegClass, MBDataKey
from monai.data import Dataset

class MBClsModel(ClsModel):
    conf: MBClsConf

    def get_cls_feature_dim(self):
        pooling_feature_size = self.backbone_dummy()[1].feature_maps[-1].shape[1]
        return pooling_feature_size * len(self.conf.pool_types)

    def create_cls_head(self):
        cls_feature_dim = self.get_cls_feature_dim()
        if self.conf.use_clinical:
            cls_feature_dim += 3
        # self.cls_head = nn.Sequential(
        #     nn.Linear(pooling_feature_size * len(conf.pool_types), pooling_feature_size),
        #     nn.ReLU(),
        #     nn.Linear(pooling_feature_size, conf.num_cls_classes),
        # )
        return nn.Linear(cls_feature_dim, self.conf.num_cls_classes)

    def __init__(self, conf: MBClsConf):
        super().__init__(conf)
        self.cls_head = self.create_cls_head()
        init_common(self.cls_head)

    @property
    def flip_keys(self):
        return [DataKey.IMG, *self.conf.pred_keys]

    @property
    def cls_names(self):
        return get_cls_names(self.conf.cls_scheme)

    def cal_feature(self, batch: dict):
        conf = self.conf
        feature_map = self.backbone.forward(batch[DataKey.IMG]).feature_maps[-1]
        features = []
        for pool_type in self.conf.pool_types:
            if pool_type not in list(SegClass):
                raise ValueError
            mask = batch[f'{pool_type}-pred']
            pooling_mask = einops.reduce(mask, 'n 1 (h s0) (w s1) (d s2) -> n 1 h w d', 'sum', **{
                f's{i}': s
                for i, s in enumerate(conf.pooling_level_stride)
            }) >= conf.pooling_th
            match conf.pooling_layer:
                case 'avg':
                    feature = torch.masked.mean(feature_map, dim=(2, 3, 4), mask=pooling_mask)
                case 'max':
                    feature = torch.masked.amax(feature_map, dim=(2, 3, 4), mask=pooling_mask)
                case _:
                    raise ValueError
            features.append(feature)

        feature = torch.cat(features, dim=1)
        return feature

    def cal_logit_impl(self, batch: dict):
        feature = self.cal_feature(batch)
        if self.conf.use_clinical:
            feature = torch.cat([feature, batch[MBDataKey.CLINICAL]], dim=1)
        logit = self.cls_head(feature)
        return logit

class MBM2FClsModel(MBClsModel, Mask2Former):
    conf: MBM2FClsConf
    mask_to_cls: torch.Tensor
    cls_to_mask: torch.Tensor

    def __init__(self, conf: MBM2FClsConf):
        super().__init__(conf)
        self.register_buffer('mask_to_cls', torch.empty(conf.num_fg_classes, dtype=torch.long))
        self.register_buffer('cls_to_mask', torch.empty(conf.num_fg_classes, dtype=torch.long))

    def get_cls_feature_dim(self):
        return self.transformer_decoder.hidden_dim

    # def get_param_groups(self) -> list[NamedParamGroup]:
    #     conf = self.conf
    #     param_groups = super().get_param_groups()
    #     cls_head_groups = []
    #     # yes, please refactor this in the future if possible
    #     no_weight_decay_keys = get_no_weight_decay_keys(self)
    #     for param_group in param_groups:
    #         cls_head_group: NamedParamGroup = {
    #             'param_names': [],
    #             'lr': conf.cls_head_optim.lr,
    #             'weight_decay': conf.cls_head_optim.weight_decay if param_group['param_names'][0] in no_weight_decay_keys else 0
    #         }
    #         params = param_group.pop('params', None)
    #         remained_idx = []
    #         for i, name in enumerate(param_group['param_names']):
    #             if name.startswith('cls_head.'):
    #                 cls_head_group['param_names'].append(name)
    #                 if params is not None:
    #                     cls_head_group.setdefault('params', []).append(params[i])
    #             else:
    #                 remained_idx.append(i)
    #         if len(cls_head_group['param_names']) > 0:
    #             cls_head_groups.append(cls_head_group)
    #         param_group['param_names'] = list(cytoolz.get(remained_idx, param_group['param_names']))
    #         if params is not None:
    #             param_group['params'] = list(cytoolz.get(remained_idx, params))
    #
    #     param_groups.extend(cls_head_groups)
    #     return param_groups

    def on_fit_start(self):
        super().on_fit_start()
        from mbs.datamodule import MBM2FClsDataModule
        dm: MBM2FClsDataModule = self.trainer.datamodule
        data_loader = dm.build_eval_dataloader(Dataset(dm.train_data(), transform=dm.predict_transform()))
        all_class_probs = []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                batch = move_data_to_device(batch, self.device)
                [*_, class_logits], _ = self.forward(batch[DataKey.IMG])
                class_probs = class_logits.softmax(dim=-1)
                all_class_probs.append(class_probs[..., 1:])
        self.train()

        mean_class_probs = torch.cat(all_class_probs, dim=0).mean(dim=0)
        _, mask_to_cls = linear_sum_assignment(mean_class_probs.cpu())
        self.mask_to_cls.copy_(torch.from_numpy(mask_to_cls))
        self.cls_to_mask[self.mask_to_cls] = torch.arange(self.conf.num_fg_classes, device=self.device)

    def cal_feature(self, batch: dict):
        x = batch[DataKey.IMG]
        manual_mask = batch[DataKey.MASK][:, self.mask_to_cls]
        [*_, mask_embeddings], _ = self.forward_mask(x, manual_mask)
        tumor_embedding = mask_embeddings[:, self.cls_to_mask[list(SegClass).index(SegClass.AT)]]
        return tumor_embedding
