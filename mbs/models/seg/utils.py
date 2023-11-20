from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.types import spatial_param_t
from monai.utils import BlendMode, InterpolateMode

@dataclass
class SlidingWindowInferenceConf:
    window_size: spatial_param_t[int]
    batch_size: int = 4
    overlap: float = 0.5
    blend_mode: BlendMode = BlendMode.GAUSSIAN
    check_window_size: bool = True

def binary_kl_div(logit_p: torch.Tensor, logit_q: torch.Tensor):
    """
    calculate binary KL divergence p * ln p/q + (1-p) * ln (1-p)/(1-q)
    """
    log_p = nnf.logsigmoid(logit_p)
    log_q = nnf.logsigmoid(logit_q)
    p = logit_p.sigmoid()
    # log_p - logit_p = log(1 - p)
    return p * (log_p - log_q) + (1 - p) * (log_p - logit_p - log_q + logit_q)

def symmetric_binary_kl_div(logit_p: torch.Tensor, logit_q: torch.Tensor):
    return binary_kl_div(logit_p, logit_q) + binary_kl_div(logit_q, logit_p)

class DeepSupervisionWrapper(nn.Module):
    weight: torch.Tensor

    def __init__(self, loss: nn.Module, num_ds: int, mode: InterpolateMode = InterpolateMode.NEAREST_EXACT):
        super().__init__()
        self.loss = loss
        weight = torch.tensor([1 / (1 << i) for i in range(num_ds)])
        weight /= weight.sum()
        self.register_buffer('weight', weight)
        self.mode = mode

    @staticmethod
    @lru_cache(1)
    def prepare_labels(label: torch.Tensor, spatial_shapes: tuple[torch.Size, ...], mode: InterpolateMode) -> list[torch.Tensor]:
        label_shape = label.shape[2:]
        return [
            nnf.interpolate(label, shape, mode=mode) if label_shape != shape else label
            for shape in spatial_shapes
        ]

    def forward(self, deep_logits: list[torch.Tensor], label: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        spatial_shapes = tuple([logits.shape[2:] for logits in deep_logits])
        deep_labels = self.prepare_labels(label, spatial_shapes, self.mode)
        ds_losses = [
            self.loss(logits, deep_label)
            for logits, deep_label in zip(deep_logits, deep_labels)
        ]
        ds_loss = torch.dot(self.weight, torch.stack(ds_losses))
        return ds_loss, ds_losses
