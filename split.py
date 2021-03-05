# split patients into some folds

import json
import random

import nibabel as nib
import pandas as pd
from tqdm import tqdm

from preprocess import output_dir as processed_dir

patients = {}
n_folds = 4
folds = [[] for _ in range(n_folds)]
random.seed(233)

patient_info = pd.read_csv('patient_info.csv')

for _, (patient, sex, age, weight, subgroup) in tqdm(patient_info.iterrows(), ncols=80, total=patient_info.shape[0]):
    scans = list(filter(lambda scan: nib.load(scan).get_fdata().ndim == 3, (processed_dir / patient).glob('*.nii.gz')))
    if len(scans) < 3:
        print(patient)
        continue
    patients.setdefault(subgroup, []).append({
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

json.dump(folds, open('folds.json', 'w'), indent=4, ensure_ascii=False)
