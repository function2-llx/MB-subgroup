import einops
import torch
from torch.nn import functional as nnf

from luolib.models.lightning import Mask2Former, SegInferer
from luolib.utils import DataKey
from mbs.utils.enums import SegClass
from monai.metrics import DiceMetric
from monai.utils import MetricReduction

from mbs.conf import MBM2FConf

class MBM2FModel(Mask2Former, SegInferer):
    conf: MBM2FConf

    def __init__(self, conf):
        super().__init__(conf)

        self.val_metrics = {
            'dice': DiceMetric(include_background=True),
        }

    def seg_predictor(self, x: torch.Tensor):
        [*_, class_logits], [*_, mask_logits] = self.forward(x)
        class_probs = class_logits.softmax(dim=-1)[..., 1:]
        # for semantic segmentation and related tasks, we assume that nq = c, and queries and classes have perfect match
        class_probs /= class_probs.sum(dim=1, keepdim=True)
        mask_probs = mask_logits.sigmoid()
        seg = einops.einsum(class_probs, mask_probs, 'n nq c, n nq ... -> n c ...')
        seg = nnf.interpolate(seg, x.shape[2:], mode=self.interpolate_mode)
        return seg

    def training_step(self, batch):
        # img = batch[DataKey.IMG]
        # mask_labels = batch[DataKey.SEG]
        # from luolib.utils import IndexTracker
        # from mbs.utils.enums import Modality
        # for i in range(img.shape[0]):
        #     seg = mask_labels[i]
        #     for j in range(seg.shape[0]):
        #         IndexTracker(
        #             img[i, list(Modality).index(Modality.T2)].cpu().numpy(),
        #             seg[j].cpu().numpy(),
        #             zyx=True,
        #         )
        return super().training_step(batch)

    def on_validation_epoch_start(self):
        if self.conf.val_empty_cuda_cache:
            torch.cuda.empty_cache()
        for metric in self.val_metrics.values():
            metric.reset()

    def validation_step(self, batch, *args, **kwargs):
        img = batch[DataKey.IMG]
        seg = batch[DataKey.SEG]
        probs = self.sw_infer(img)
        probs = nnf.interpolate(probs, seg.shape[2:], mode=self.interpolate_mode)
        pred = (probs > 0.5).long()
        for k, metric in self.val_metrics.items():
            metric(pred, seg)

    def on_validation_epoch_end(self) -> None:
        if self.conf.val_empty_cuda_cache:
            torch.cuda.empty_cache()

        for name, metric in self.val_metrics.items():
            m = metric.aggregate(reduction=MetricReduction.MEAN_BATCH)
            for i in range(m.shape[0]):
                class_name = list(SegClass)[i]
                self.log(f'val/{name}/{class_name}', m[i], sync_dist=True)
            avg = m.nanmean()
            self.log(f'val/{name}/avg', avg, sync_dist=True)
