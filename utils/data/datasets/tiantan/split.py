# split patients into some folds

import json
import random
from collections import OrderedDict

patients = {}
n_folds = 3
folds = [[] for _ in range(n_folds)]
random.seed(233)

if __name__ == '__main__':
    cohort = json.load(open('cohort.json'))
    patients = OrderedDict([(group, [patient for patient in cohort if patient['subgroup'] == group]) for group in ['WNT', 'SHH', 'G3', 'G4']])

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
