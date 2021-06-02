import json
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map
import monai.transforms as monai_transforms

from utils.data import MultimodalDataset
from utils.transforms import RandSampleSlicesD


class ConvertToMultiChannelBasedOnBratsClassesd(monai_transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
        - the GD-enhancing tumor (ET — label 4)
        - the peritumoral edema (ED — label 2)
        - and the necrotic and non-enhancing tumor core (NCR/NET — label 1)

    # label 1 is the peritumoral edema
    # label 2 is the GD-enhancing tumor
    # label 3 is the necrotic and non-enhancing tumor core
    # The possible classes are TC (Tumor core), WT (Whole tumor)
    # and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge labels 1, 2 and 4 to construct WT
            result.append(d[key] != 0)
            # merge label 1 and label 4 to construct TC
            result.append((d[key] == 1) | (d[key] == 4))
            # label 4 is AT
            result.append(d[key] == 4)
            d[key] = np.concatenate(result, axis=0).astype(np.float32)
        return d

data_dir = Path(__file__).parent / 'processed'

def load_subject(subject):
    subject, info = subject
    data = dict(np.load(data_dir / f'{subject}/data.npz'))
    assert 'img' in data and 'seg' in data
    data['label'] = {
        'LGG': 0,
        'HGG': 1,
    }[info['grade']]
    return data

# load data for pre-training
def load_all(args) -> MultimodalDataset:
    subjects = list(json.load(open(data_dir / 'subjects.json')).items())
    if args.debug:
        subjects = subjects[:30]
    data = process_map(
        load_subject,
        subjects,
        desc='loading BraTS20',
        ncols=80,
    )
    # intensity of images were normalized during pre-processing
    transforms = [
        ConvertToMultiChannelBasedOnBratsClassesd(keys='seg'),
        RandSampleSlicesD('img', args.sample_slices)
    ]
    if 'crop' in args.aug:
        transforms.extend([
            monai_transforms.RandSpatialCropD(
                keys='img',
                roi_size=(args.sample_size, args.sample_size, args.sample_slices),
                random_center=False,
                random_size=True,
            ),
        ])
    transform = monai_transforms.Compose([
        # randomly select slices
        RandSampleSlicesD(['img', 'seg'], num_slices=args.sample_slices),
        monai_transforms.RandSpatialCropd(
            keys=['img', 'seg'],
            roi_size=(args.sample_size, args.sample_size, args.sample_slices),
            random_size=True,
            random_center=False,
        ),
        # the other day I says (may outdated):
        # note: this one seems not work, harmful to pretrain
        # 155: slices of BraTS; 24: slices of Tiantan; 155 // 24 = 6
        monai_transforms.RandFlipd(keys='img', prob=0.5, spatial_axis=0),
        monai_transforms.RandFlipd(keys='img', prob=0.5, spatial_axis=1),
        monai_transforms.RandFlipd(keys='img', prob=0.5, spatial_axis=2),
        monai_transforms.RandRotate90d(keys='img', prob=0.5, max_k=1),
        monai_transforms.NormalizeIntensityd(keys='img', nonzero=True, channel_wise=True),
        monai_transforms.RandScaleIntensityd(keys='img', factors=0.1, prob=0.5),
        monai_transforms.RandShiftIntensityd(keys='img', offsets=0.1, prob=0.5),
        monai_transforms.Resized(
            keys=['img', 'seg'],
            spatial_size=(args.sample_size, args.sample_size, args.sample_slices),
            mode=('area', 'nearest'),
        ),
        monai_transforms.ToTensord(keys=['img', 'seg']),
    ])

    return MultimodalDataset(data, transform, num_classes=None, progress=True)
