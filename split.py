# split patients into some folds

import os
import random
import json
from glob import glob

import pandas as pd

processed_dir = 'processed_3d'
patients = {}
n_folds = 4
folds = [[] for _ in range(n_folds)]
random.seed(233)

n_scans = []
for _, (patient, subgroup) in pd.read_csv('subgroup.csv').iterrows():
    filenames = glob(os.path.join(processed_dir, patient, '*.nii.gz'))
    num = len(filenames)
    assert num
    if num < 3:
        print(patient)
        continue
    n_scans.append(num)
    patients.setdefault(subgroup, []).append(patient)

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
