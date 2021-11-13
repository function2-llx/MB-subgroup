import logging
import random
from abc import ABC
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from transformers import TrainingArguments

from monai import transforms as monai_transforms
from monai.transforms import Transform
from utils.args import DataTrainingArgs, ModelArgs, FinetuneArgs
from utils.transforms import RandSampleSlicesD

class RunnerBase(ABC):
    def __init__(self, args: Union[TrainingArguments, DataTrainingArgs, ModelArgs]):
        self.args = args
        self.setup_logging()
        if args.do_train:
            self.set_determinism()

    def combine_loss(self, cls_loss, seg_loss, vae_loss=None) -> torch.FloatTensor:
        ret = self.args.cls_factor * cls_loss + self.args.seg_factor * seg_loss
        if vae_loss is not None:
            ret += vae_loss * self.args.vae_factor
        return ret

    def set_determinism(self):
        seed = 2333
        # harm the speed
        # monai.utils.set_determinism(seed)
        # if self.is_world_master():
        logging.info(f'set random seed of {seed}\n')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # not supported currently, throw RE for AdaptiveAvgPool
        # torch.use_deterministic_algorithms(True)

    def is_world_master(self) -> bool:
        return self.args.process_index == 0

    def setup_logging(self):
        args = self.args
        handlers = [logging.StreamHandler()]
        if args.do_train:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(output_dir / 'train.log', mode='a'))
            logging.basicConfig(
                format='%(asctime)s [%(levelname)s] %(message)s',
                datefmt=logging.Formatter.default_time_format,
                level=logging.INFO,
                handlers=handlers,
                force=True,
            )

    @staticmethod
    def get_train_transforms(args: DataTrainingArgs) -> List[Transform]:
        from monai.utils import InterpolateMode

        keys = ['img', 'seg']
        resize_mode = [InterpolateMode.AREA, InterpolateMode.NEAREST]
        # train_transforms = []
        train_transforms = [
            RandSampleSlicesD(keys=keys, num_slices=args.sample_slices),
        ]
        if 'crop' in args.aug:
            train_transforms.extend([
                monai_transforms.RandSpatialCropD(
                    keys=keys,
                    roi_size=(args.sample_size, args.sample_size, args.sample_slices),
                    random_center=False,
                    random_size=True,
                ),
            ])
        if 'flip' in args.aug:
            train_transforms.extend([
                monai_transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                monai_transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                monai_transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                monai_transforms.RandRotate90d(keys=keys, prob=0.5, max_k=1),
            ])
        if 'voxel' in args.aug:
            # seg won't be affected
            train_transforms.extend([
                monai_transforms.RandScaleIntensityd(keys='img', factors=0.1, prob=0.5),
                monai_transforms.RandShiftIntensityd(keys='img', offsets=0.1, prob=0.5),
            ])
        # train_transforms.append(monai_transforms.ToTensorD(keys))
        train_transforms.extend([
            monai_transforms.ResizeD(
                keys=keys,
                spatial_size=(args.sample_size, args.sample_size, args.sample_slices),
                mode=resize_mode,
            ),
            monai_transforms.ToTensorD(keys=keys),
        ])

        return train_transforms

    @staticmethod
    def get_inference_transforms(args: FinetuneArgs):
        keys = ['img', 'seg']
        # resize_mode = [InterpolateMode.AREA]
        #     keys.append('seg')
            # resize_mode.append(InterpolateMode.NEAREST)

        return [
            # SampleSlicesD(keys, 2, args.sample_slices),
            # monai_transforms.ResizeD(
            #     keys,
            #     spatial_size=(args.sample_size, args.sample_size, args.sample_slices),
            #     mode=resize_mode,
            # ),
            monai_transforms.ToTensorD(keys),
        ]
