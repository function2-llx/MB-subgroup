import json
from pathlib import Path

import numpy as np
from monai import transforms as monai_transforms
import pandas as pd
from tqdm.contrib.concurrent import process_map

from utils.dicom_utils import ScanProtocol
from utils.transforms import ConvertToMultiChannelBasedOnBratsClassesd

labels = pd.read_csv('origin/train_labels.csv', dtype={'BraTS21ID': str})
labels.set_index('BraTS21ID', inplace=True)

def parse_modality(modality):
    return {
        't1': ScanProtocol.T1,
        't1ce': ScanProtocol.T1c,
        't2': ScanProtocol.T2,
        'seg': 'seg',
    }[modality]

def load_subject(subject: str):
    subject_path = Path('origin') / subject
    output_dir = Path('processed') / subject
    output_dir.mkdir(parents=True, exist_ok=True)
    data = loader({
        parse_modality(modality): str(subject_path / f'{subject}_{modality}.nii.gz')
        for modality in ['t1', 't1ce', 't2', 'seg']
    })
    np.savez(output_dir / 'data.npz', **data)
    subject_info = {'MGMT': -100}
    if subject[-5:] in labels.index:
        subject_info['MGMT'] = int(labels.loc[subject[-5:], 'MGMT_value'])
    return subject_info

def main():
    subjects = []
    for subject_path in Path('origin').iterdir():
        if not subject_path.is_dir():
            continue
        subject = subject_path.name
        if not subject.startswith('BraTS2021'):
            continue
        subjects.append(subject)

    global loader
    loader = monai_transforms.Compose([
        monai_transforms.LoadImaged(list(ScanProtocol) + ['seg']),
        monai_transforms.AddChanneld(list(ScanProtocol) + ['seg']),
        monai_transforms.Orientationd(list(ScanProtocol) + ['seg'], 'LAS'),
        monai_transforms.ConcatItemsd(list(ScanProtocol), 'img'),
        monai_transforms.SelectItemsd(['img', 'seg']),
        monai_transforms.ThresholdIntensityD('img', threshold=0),
        monai_transforms.NormalizeIntensityD('img', channel_wise=True, nonzero=True),
        ConvertToMultiChannelBasedOnBratsClassesd('seg'),
    ])
    subjects = {
        subject: info
        for subject, info in zip(
            subjects,
            process_map(load_subject, subjects, ncols=80, max_workers=16),
        )
    }
    json.dump(subjects, open('processed/subjects.json', 'w'), indent=4, ensure_ascii=False)
    pd.DataFrame(subjects).transpose().to_excel('processed/subjects.xlsx')

if __name__ == "__main__":
    main()
