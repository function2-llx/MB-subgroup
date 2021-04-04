import json
from pathlib import Path

import numpy as np
from monai.transforms import *
from tqdm.contrib.concurrent import process_map

from utils.data import MultimodalDataset
from utils.transforms import RandSampleSlicesd

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
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
        max_workers=16,
    )
    transform = Compose([
        ConvertToMultiChannelBasedOnBratsClassesd('seg'),
        ThresholdIntensityd('img', threshold=0),
        Resized(
            keys=('img', 'seg'),
            spatial_size=(args.sample_size, args.sample_size, -1),
            mode=('area', 'nearest'),
        ),
        # 155: slices of BraTS; 24: slices of Tiantan; 155 // 24 = 6
        RandSampleSlicesd(('img', 'seg'), sample_slices=args.sample_slices, spacing=155 // 24),
        RandFlipd(keys=['img', 'seg'], prob=0.5, spatial_axis=0),
        NormalizeIntensityd('img', nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys='img', factors=0.1, prob=0.5),
        RandShiftIntensityd(keys='img', offsets=0.1, prob=0.5),
        ToTensord(keys=['img', 'seg']),
    ])

    return MultimodalDataset(data, transform, num_classes=None)
