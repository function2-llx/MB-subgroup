# split patients into some folds

import json
import random

import numpy as np
import pandas as pd

from finetuner_base import FinetuneArgs
from utils.argparse import ArgParser

def main():
    args, = ArgParser([FinetuneArgs]).parse_args_into_dataclasses()
    args: FinetuneArgs
    n_folds = args.n_folds
    folds = [[] for _ in range(n_folds)]
    subgroups = args.subgroups
    random.seed(2333)
    cohort = pd.read_excel('cohort.xlsx')
    # group by subtypes
    groups = [cohort[cohort['subgroup'] == subgroup]['name(raw)'].values for subgroup in subgroups]

    counts = np.empty((len(subgroups), n_folds), dtype=int)
    for subgroup_id, patients in enumerate(groups):
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

    json.dump(folds, open(args.folds_file, 'w'), indent=4, ensure_ascii=False)
    # df = pd.DataFrame(counts, columns=[f'第{i}折' for i in range(1, 4)], index=subgroups)

if __name__ == '__main__':
    main()
