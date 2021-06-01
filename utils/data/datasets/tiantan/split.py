# split patients into some folds

import json
import random
from collections import OrderedDict

import numpy as np
import pandas as pd

grouped_patients = {}
n_folds = 3
folds = [[] for _ in range(n_folds)]
random.seed(233)
subgroups = ['WNT', 'SHH', 'G3', 'G4']

if __name__ == '__main__':
    cohort = json.load(open('cohort.json'))
    # group by subtypes
    grouped_patients = [[patient for patient in cohort if patient['subgroup'] == group] for group in subgroups]
    counts = np.empty((len(subgroups), n_folds), dtype=int)
    for subgroup_id, patients in enumerate(grouped_patients):
        q, r = divmod(len(patients), n_folds)
        assert q >= 1
        random.shuffle(patients)
        chunks = [q] * (n_folds - r) + [q + 1] * r
        random.shuffle(chunks)
        for fold_id in range(1, len(chunks)):
            chunks[fold_id] += chunks[fold_id - 1]
        chunks = [0] + chunks

        for fold_id, (l, r) in enumerate(zip(chunks, chunks[1:])):
            folds[fold_id].extend(patients[l:r])
            counts[subgroup_id, fold_id] = r - l

    print([len(fold) for fold in folds])

    json.dump(folds, open(f'folds-{n_folds}.json', 'w'), indent=4, ensure_ascii=False)
    df = pd.DataFrame(counts, columns=[f'第{i}折' for i in range(1, 4)], index=subgroups)
