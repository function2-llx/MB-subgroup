import numpy as np
import pandas as pd

from luolib.utils import DataSplit
from mbs.datamodule import load_merged_plan
from monai.data import partition_dataset_classes, partition_dataset

from mbs.utils.enums import MBDataKey, MBGroup, PROCESSED_DIR, SUBGROUPS

np.random.seed(2333)

test_sizes = {
    'WNT': 10,
    'SHH': 18,
    'G3': 12,
    'G4': 20,
}

no_seg = ['483517', '586779']

def main():
    plan = load_merged_plan()
    child_idx = plan['group'] == MBGroup.CHILD
    child_plan = plan[child_idx]
    split_plan = pd.Series(index=plan.index.sort_values(), name='split')
    for subgroup, cnt in test_sizes.items():
        subgroup_idx = child_plan.index[child_plan[MBDataKey.SUBGROUP] == subgroup]
        force_test_mask = subgroup_idx.isin(no_seg)
        cnt -= force_test_mask.sum()
        for case in subgroup_idx[force_test_mask]:
            split_plan[case] = DataSplit.TEST
        subgroup_idx = subgroup_idx[~force_test_mask]
        for case in np.random.choice(subgroup_idx, cnt, replace=False):
            split_plan[case] = DataSplit.TEST
    train_plan = plan[split_plan != DataSplit.TEST]
    num_folds = 5
    cv_parts = [[] for _ in range(num_folds)]
    for subgroup in SUBGROUPS:
        subgroup_idx = train_plan.index[train_plan[MBDataKey.SUBGROUP] == subgroup]
        rem_mask = subgroup_idx.isin(np.random.choice(subgroup_idx, subgroup_idx.size % num_folds, replace=False))
        for i, part in enumerate(partition_dataset(subgroup_idx[~rem_mask], num_partitions=num_folds, shuffle=True)):
            cv_parts[i].extend(part)
        rem_idx = subgroup_idx[rem_mask]
        while rem_idx.size > 0:
            min_len = min(map(len, cv_parts))
            folds = [i for i in range(num_folds) if len(cv_parts[i]) == min_len]
            for i, fold in enumerate(np.random.choice(folds, cnt := min(len(folds), rem_idx.size), replace=False)):
                cv_parts[fold].append(rem_idx[i])
            rem_idx = rem_idx[cnt:]

    for i, part in enumerate(cv_parts):
        for case in part:
            split_plan[case] = i

    split_plan.to_excel(PROCESSED_DIR / 'split.xlsx', freeze_panes=(1, 0))

if __name__ == '__main__':
    main()
