from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

__all__ = ["Backbone"]

from transformers import TrainingArguments

class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, permute: bool):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        pass

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0

    def finetune_parameters(self, args: TrainingArguments):
        params = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        no_decay = ['bias', 'Norm.weight']
        grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay,
             'lr': args.learning_rate},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.learning_rate},
        ]
        return grouped_parameters

    # def output_shape(self):
    #     """
    #     Returns:
    #         dict[str->ShapeSpec]
    #     """
    #     # this is a backward-compatible default
    #     return {
    #         name: ShapeSpec(
    #             channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
    #         )
    #         for name in self._out_features
    #     }
