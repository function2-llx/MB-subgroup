import einops
import torch
from torch import nn

from luolib.models import ClsModel
from luolib.utils import DataKey

from mbs.conf import MBClsConf
from mbs.datamodule import MBClsDataModule
from mbs.utils.enums import SegClass, SUBGROUPS

class MBClsModel(ClsModel):
    conf: MBClsConf

    def __init__(self, conf: MBClsConf):
        super().__init__(conf)
        pooling_feature_size = self.backbone_dummy()[1].feature_maps[-1].shape[1]
        self.cls_head = nn.Sequential(
            nn.Linear(pooling_feature_size * len(conf.pool_types), pooling_feature_size),
            nn.Tanh(),
            nn.Linear(pooling_feature_size, conf.num_cls_classes),
        )

    @property
    def flip_keys(self):
        return [DataKey.IMG, *MBClsDataModule.pred_keys]

    @property
    def cls_names(self):
        return SUBGROUPS

    def cal_logit_impl(self, batch: dict):
        conf = self.conf
        feature_map = self.backbone.forward(batch[DataKey.IMG]).feature_maps[-1]
        features = []
        for pool_type in self.conf.pool_types:
            if pool_type in list(SegClass):
                mask = batch[f'{pool_type}-pred']
                pooling_mask = einops.reduce(mask, 'n 1 (h s0) (w s1) (d s2) -> n 1 h w d', 'sum', **{
                    f's{i}': s
                    for i, s in enumerate(conf.pooling_level_stride)
                }) >= conf.pooling_th
                feature = torch.masked.mean(feature_map, dim=(2, 3, 4), mask=pooling_mask)
                features.append(feature)
            else:
                raise ValueError

        feature = torch.cat(features, dim=1)
        logit = self.cls_head(feature)
        return logit
