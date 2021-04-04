import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from monai.transforms import *
from tqdm.contrib.concurrent import process_map

from utils.dicom_utils import ScanProtocol
from utils.transforms import ConcatItemsAllowSingled


def parse_modality(modality):
    return {
        't1': ScanProtocol.T1,
        't1ce': ScanProtocol.T1c,
        't2': ScanProtocol.T2,
        'seg': 'seg',
    }[modality]

def load_subject(subject):
    subject_path = Path('origin') / subject
    output_dir = Path('processed') / subject
    output_dir.mkdir(parents=True, exist_ok=True)
    data = loader({
        parse_modality(modality): str(subject_path / f'{subject}_{modality}.nii.gz')
        for modality in ['t1', 't1ce', 't2', 'seg']
    })
    np.savez_compressed(output_dir / 'data.npz', **data)
    subject_info = {
        'grade': mapping.loc[subject, 'Grade'],
    }
    if subject in survival_info.index:
        subject_survival_info = survival_info.loc[subject]
        survival = subject_survival_info['Survival_days']
        if not survival.startswith('ALIVE'):
            subject_info.update({
                'age': subject_survival_info['Age'],
                'survival': int(survival),
            })
            resect = subject_survival_info['Extent_of_Resection']
            if isinstance(resect, str):
                subject_info.update({'resect': resect})
    return subject_info

if __name__ == "__main__":
    mapping = pd.read_csv('origin/name_mapping.csv', index_col='BraTS_2020_subject_ID')
    survival_info = pd.read_csv('origin/survival_info.csv', index_col='Brats20ID')
    loader = Compose([
        LoadImaged(list(ScanProtocol) + ['seg']),
        AddChanneld(list(ScanProtocol) + ['seg']),
        Orientationd(list(ScanProtocol) + ['seg'], 'PLI'),
        ConcatItemsAllowSingled(list(ScanProtocol), 'img'),
        SelectItemsd(['img', 'seg']),
    ])
    subjects = []
    for subject_path in Path('origin').iterdir():
        subject = subject_path.parts[-1]
        if subject.startswith('BraTS20_Training_'):
            subjects.append(subject)
            continue
    subjects = {
        subject: info
        for subject, info in zip(
            subjects,
            process_map(load_subject, subjects, ncols=80, max_workers=16),
        )
    }
    json.dump(subjects, open('processed/subjects.json', 'w'), indent=4, ensure_ascii=False)
