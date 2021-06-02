import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import monai.transforms as monai_transforms
from monai import transforms as monai_transforms
from tqdm.contrib.concurrent import process_map

from utils.dicom_utils import ScanProtocol


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

if __name__ == "__main__":
    mapping = pd.read_csv('origin/name_mapping.csv', index_col='BraTS_2020_subject_ID')
    survival_info = pd.read_csv('origin/survival_info.csv', index_col='Brats20ID')
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
