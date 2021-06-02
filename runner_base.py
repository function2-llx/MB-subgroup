import logging
from abc import ABC

import monai
from monai import transforms as monai_transforms

from utils.transforms import RandSampleSlicesD


class RunnerBase(ABC):
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        if self.args.train:
            self.set_determinism()

    def set_determinism(self):
        seed = self.args.seed
        monai.utils.set_determinism(seed)
        if self.is_world_master():
            logging.info(f'set random seed of {seed}\n')
        # not supported currently, throw RE for AdaptiveAvgPool
        # torch.use_deterministic_algorithms(True)

    def is_world_master(self) -> bool:
        return self.args.rank == 0

    def setup_logging(self):
        args = self.args
        handlers = [logging.StreamHandler()]
        if args.train:
            args.model_output_root.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(args.model_output_root / 'train.log', mode='a'))
            logging.basicConfig(
                format='%(asctime)s [%(levelname)s] %(message)s',
                datefmt=logging.Formatter.default_time_format,
                level=logging.INFO,
                handlers=handlers,
                force=True,
            )

    def get_train_transforms(self, with_seg=True):
        from monai.utils import InterpolateMode

        keys = ['img']
        resize_mode = [InterpolateMode.AREA]
        if with_seg:
            keys.append('seg')
            resize_mode.append(InterpolateMode.NEAREST)
        train_transforms = [
            RandSampleSlicesD(keys=keys, num_slices=self.args.sample_slices),
        ]
        if 'crop' in self.args.aug:
            train_transforms.extend([
                monai_transforms.RandSpatialCropD(
                    keys=keys,
                    roi_size=(self.args.sample_size, self.args.sample_size, self.args.sample_slices),
                    random_center=False,
                    random_size=True,
                ),
            ])
        if 'flip' in self.args.aug:
            train_transforms.extend([
                monai_transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                monai_transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                monai_transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                monai_transforms.RandRotate90d(keys=keys, prob=0.5, max_k=1),
            ])
        if 'voxel' in self.args.aug:
            # seg won't be affected
            train_transforms.extend([
                monai_transforms.RandScaleIntensityd(keys='img', factors=0.1, prob=0.5),
                monai_transforms.RandShiftIntensityd(keys='img', offsets=0.1, prob=0.5),
            ])
        train_transforms.extend([
            monai_transforms.ResizeD(
                keys=keys,
                spatial_size=(self.args.sample_size, self.args.sample_size, self.args.sample_slices),
                mode=resize_mode,
            ),
            monai_transforms.ToTensorD(keys=keys),
        ])

        return train_transforms
