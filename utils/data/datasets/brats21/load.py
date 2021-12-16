import json
from pathlib import Path
from typing import List, Dict

import numpy as np

from utils.args import DataTrainingArgs

data_dir = Path(__file__).parent / 'preprocessed'

# def load_subject(subject):
#     subject, info = subject
#     data = dict(np.load(str(data_dir / subject' / 'data.npz')))
#     assert 'img' in data and 'seg' in data
#     data['label'] = info['MGMT']
#     return data

# load data for pre-training
def load_all(args: DataTrainingArgs) -> List[Dict]:
    ret = [
        {
            'data': str(data_dir / f'{subject}/data.npz'),
            'label': info['MGMT'],
        }
        for subject, info in json.load(open(data_dir / 'subjects.json')).items()
    ]
    if args.subjects is not None:
        ret = ret[:args.subjects]
    return ret
    # subjects = list(.items())
    # if args.subjects is not None:
    #     subjects = subjects[:args.subjects]
    # RAM supports you, say thank you, RAM
    # data = process_map(
    #     load_subject,
    #     subjects,
    #     desc='loading BraTS21',
    #     ncols=80,
    #     max_workers=32,
    # )
    # return data
