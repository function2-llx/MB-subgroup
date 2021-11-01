import logging
from abc import ABC

import monai
from monai import transforms as monai_transforms
from monai.utils import InterpolateMode

from utils.conf import Conf
from utils.transforms import RandSampleSlicesD, SampleSlicesD

class RunnerBase(ABC):
    def __init__(self, conf: Conf):
        self.conf = conf
        self.setup_logging()
        # if conf.do_train:
        #     self.set_determinism()

    # def set_determinism(self):
    #     seed = self.conf.seed
    #     monai.utils.set_determinism(seed)
    #     if self.is_world_master():
    #         logging.info(f'set random seed of {seed}\n')
        # not supported currently, throw RE for AdaptiveAvgPool
        # torch.use_deterministic_algorithms(True)

    def is_world_master(self) -> bool:
        return self.conf.rank == 0

    def setup_logging(self):
        conf = self.conf
        handlers = [logging.StreamHandler()]
        if conf.do_train:
            conf.model_output_root.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(conf.model_output_root / 'train.log', mode='a'))
            logging.basicConfig(
                format='%(asctime)s [%(levelname)s] %(message)s',
                datefmt=logging.Formatter.default_time_format,
                level=logging.INFO,
                handlers=handlers,
                force=True,
            )

    @staticmethod
    def get_train_transforms(conf: Conf):
        from monai.utils import InterpolateMode

        keys = ['img', 'seg']
        # resize_mode = [InterpolateMode.AREA, InterpolateMode.NEAREST]
        train_transforms = []
        # train_transforms = [
        #     RandSampleSlicesD(keys=keys, num_slices=conf.sample_slices),
        # ]
        if 'crop' in conf.aug:
            train_transforms.extend([
                monai_transforms.RandSpatialCropD(
                    keys=keys,
                    roi_size=(conf.sample_size, conf.sample_size, conf.sample_slices),
                    random_center=False,
                    random_size=True,
                ),
            ])
        if 'flip' in conf.aug:
            train_transforms.extend([
                monai_transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                monai_transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                monai_transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                monai_transforms.RandRotate90d(keys=keys, prob=0.5, max_k=1),
            ])
        if 'voxel' in conf.aug:
            # seg won't be affected
            train_transforms.extend([
                monai_transforms.RandScaleIntensityd(keys='img', factors=0.1, prob=0.5),
                monai_transforms.RandShiftIntensityd(keys='img', offsets=0.1, prob=0.5),
            ])
        train_transforms.append(monai_transforms.ToTensorD(keys))
        # train_transforms.extend([
        #     monai_transforms.ResizeD(
        #         keys=keys,
        #         spatial_size=(conf.sample_size, conf.sample_size, conf.sample_slices),
        #         mode=resize_mode,
        #     ),
        #     monai_transforms.ToTensorD(keys=keys),
        # ])

        return train_transforms

    @staticmethod
    def get_inference_transforms(conf: Conf):
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
