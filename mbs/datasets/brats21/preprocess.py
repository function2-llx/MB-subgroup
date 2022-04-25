import json
from pathlib import Path

import monai
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from mbs.utils.dicom_utils import ScanProtocol

labels = pd.read_csv('origin/train_labels.csv', dtype={'BraTS21ID': str})
labels.set_index('BraTS21ID', inplace=True)

def parse_modality(modality):
    return {
        't1': ScanProtocol.T1,
        't1ce': ScanProtocol.T1c,
        't2': ScanProtocol.T2,
        'seg': 'seg',
    }[modality]

output_dir = Path('preprocessed')

def load_subject(subject: str):
    subject_path = Path('origin') / subject
    subject_output_dir = output_dir / subject
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    data = loader({
        parse_modality(modality): str(subject_path / f'{subject}_{modality}.nii.gz')
        for modality in ['t1', 't1ce', 't2', 'seg']
    })
    np.savez(subject_output_dir / 'data.npz', **data)
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
    all_keys = [*ScanProtocol, 'seg']
    loader = monai.transforms.Compose([
        monai.transforms.LoadImaged(all_keys),
        monai.transforms.AddChanneld(all_keys),
        monai.transforms.Orientationd(all_keys, 'LAS'),
        monai.transforms.ConcatItemsd([*ScanProtocol], 'img'),
        monai.transforms.ThresholdIntensityD('img', threshold=0),
        monai.transforms.CropForegroundD(['img', 'seg'], source_key='img'),
        # monai.transforms.NormalizeIntensityD('img', channel_wise=True, nonzero=True),
        monai.transforms.ConvertToMultiChannelBasedOnBratsClassesD('seg'),
        monai.transforms.SelectItemsd(['img', 'seg']),
    ])
    subjects = {
        subject: info
        for subject, info in zip(
            subjects,
            process_map(load_subject, subjects, ncols=80, max_workers=16),
        )
    }
    json.dump(subjects, open(output_dir / 'subjects.json', 'w'), indent=4, ensure_ascii=False)
    pd.DataFrame(subjects).transpose().to_excel(output_dir / 'subjects.xlsx')

if __name__ == "__main__":
    main()
