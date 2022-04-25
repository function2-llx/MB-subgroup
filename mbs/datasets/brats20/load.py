import json
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map
from typing import List, Dict

from utils.data import MultimodalDataset

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
def load_all(args) -> List[Dict]:
    subjects = list(json.load(open(data_dir / 'subjects.json')).items())
    if args.debug:
        subjects = subjects[:30]
    data = process_map(
        load_subject,
        subjects,
        desc='loading BraTS20',
        ncols=80,
    )
    return data
