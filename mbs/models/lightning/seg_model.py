import einops
import torch
from torchmetrics import Recall

from luolib.models import SegModel
from luolib.utils import DataKey
from mbs.cnn_decoder import CNNDecoder
from mbs.conf import MBSegConf
from mbs.utils.enums import MBDataKey, SegClass
import monai

class MBSegModel(SegModel):
    args: MBSegConf

    def __init__(self, conf: MBSegConf):
        super().__init__(conf)
        self.post_transform = monai.transforms.Compose([
            monai.transforms.KeepLargestConnectedComponent(is_onehot=True),
        ])
        self.test_outputs = []
        self.recall = Recall(task='binary', num_classes=conf.num_seg_classes, average=None)
        self.recall_u = Recall(task='binary', num_classes=conf.num_seg_classes, average=None)

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
