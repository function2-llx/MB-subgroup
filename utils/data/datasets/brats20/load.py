import itertools
import json
from argparse import Namespace
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from monai.transforms import *
from tqdm.contrib.concurrent import process_map

data_dir = Path(__file__).parent / 'processed'
_transforms: Optional[Transform] = None
_args: Optional[Namespace] = None
_normalizer = NormalizeIntensityd(keys='img', channel_wise=True)


def load_subject(subject):
    subject, info = subject
    label = {
        'LGG': 0,
        'HGG': 1,
    }[info['grade']]
    data = dict(np.load(data_dir / f'{subject}/data.npz'))
    assert 'img' in data and 'seg' in data
    # tumor_slices = [i for i in range(data['seg'].shape[3]) if data['seg'][0, :, :, i].max() > 0]
    # slice spacing
    scale = data['img'].shape[3] // 24

    ret = [
        _normalizer({
            'img': data['img'][:, :, :, i:i + scale * _args.sample_slices:scale],
            'seg': data['seg'][:, :, :, i:i + scale * (_args.sample_slices - 1) + 1:scale],
            'label': label,
        })
        for i in range(data['img'].shape[3] - (scale * (_args.sample_slices - 1) + 1))
    ]
    return ret

def load_all(args) -> List[Dict]:
    global _transforms, _args
    _args = args
    _transforms = Compose([
        Resized(
            keys=('img', 'seg'),
            spatial_size=(args.sample_size, args.sample_size, -1),
            mode=('area', 'nearest'),

        ),
        ThresholdIntensityd('img', threshold=0),
    ])

    return list(itertools.chain(*process_map(
        load_subject,
        json.load(open(data_dir / 'subjects.json')).items(),
        desc='loading BraTS20',
        ncols=80,
        max_workers=16,
    )))
