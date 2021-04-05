# split patients into some folds

import json
import random
import re
from collections import Counter, OrderedDict

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data.datasets.tiantan.preprocess import output_dir as processed_dir

patients = {}
n_folds = 3
folds = [[] for _ in range(n_folds)]
random.seed(233)

patient_info = pd.read_csv(processed_dir / 'patient_info.csv')
desc_reject_list = [
    'Apparent Diffusion Coefficient (mm2/s)',
    'Ax DWI 1000b',
    'Calibration',
    'diffusion',
    'ep2d_diff_3scan_trace_p2',
    'ep2d_diffusion',
    'Exponential Apparent Diffusion Coefficient',
    'FSE100-9',
    'I_t1_tse_sag_384',
    'OAx DWI Asset',
    'PU:OAx DWI Asset',
    'SE12_HiCNR',
    'TOF_3D_multi-slab',
]

if __name__ == '__main__':
    patients = OrderedDict([(group, []) for group in ['WNT', 'SHH', 'G3', 'G4']])
    for _, (patient, sex, age, weight, subgroup) in tqdm(patient_info.iterrows(), ncols=80, total=patient_info.shape[0]):
        counter = Counter()
        # group scans by shape
        grouped_scans = {}
        all_scans = []
        raw_all_scans = sorted(
            (processed_dir / patient).glob('*.nii.gz'),
            key=lambda x: int(re.match(r'\d+', x.parts[-1]).group()),
        )

        for scan in raw_all_scans:
            data: np.ndarray = nib.load(scan).get_fdata()
            info = json.load(open(re.sub(r'nii.gz$', 'json', str(scan))))
            desc = info['SeriesDescription']
            if data.ndim == 3 and 22 <= data.shape[2] <= 26 and desc not in desc_reject_list:
                grouped_scans.setdefault(data.shape, []).append(scan)
                all_scans.append(scan)

        for scans in grouped_scans.values():
            if len(scans) >= 3:
                scans = scans[:3]
                break
        else:
            if len(all_scans) >= 3:
                scans = all_scans[:3]
            else:
                print(patient)
                continue
        patients[subgroup].append({
            'patient': patient,
            'subgroup': subgroup,
            'sex': sex,
            'age': age,
            'weight': weight,
            'scans': list(map(str, scans)),
        })

    for subgroup_patients in patients.values():
        q, r = divmod(len(subgroup_patients), n_folds)
        assert q >= 1
        random.shuffle(subgroup_patients)
        chunks = [q] * (n_folds - r) + [q + 1] * r
        random.shuffle(chunks)
        for i in range(1, len(chunks)):
            chunks[i] += chunks[i - 1]
        chunks = [0] + chunks
        for i, (l, r) in enumerate(zip(chunks, chunks[1:])):
            folds[i].extend(subgroup_patients[l:r])

    print([len(fold) for fold in folds])

    json.dump(folds, open(f'folds-{n_folds}.json', 'w'), indent=4, ensure_ascii=False)
