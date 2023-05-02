import einops
import torch
from torch import nn

from luolib.models import ClsModel
from luolib.models.init import init_linear_conv
from luolib.utils import DataKey

from mbs.conf import MBClsConf, get_cls_names
from mbs.datamodule import MBClsDataModule
from mbs.utils.enums import SegClass

class MBClsModel(ClsModel):
    conf: MBClsConf

    def __init__(self, conf: MBClsConf):
        super().__init__(conf)
        pooling_feature_size = self.backbone_dummy()[1].feature_maps[-1].shape[1]
        # self.cls_head = nn.Sequential(
        #     nn.Linear(pooling_feature_size * len(conf.pool_types), pooling_feature_size),
        #     nn.ReLU(),
        #     nn.Linear(pooling_feature_size, conf.num_cls_classes),
        # )
        self.cls_head = nn.Linear(pooling_feature_size * len(conf.pool_types), conf.num_cls_classes)
        init_linear_conv(self.cls_head)

    @property
    def flip_keys(self):
        return [DataKey.IMG, *MBClsDataModule.pred_keys]

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
        logit = self.cls_head(feature)
        return logit
