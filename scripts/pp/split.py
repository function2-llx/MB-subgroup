import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mbs.datamodule import DATA_DIR, PROCESSED_DIR
from mbs.utils.enums import MBDataKey, MBGroup
from monai.data import partition_dataset_classes
from luolib.utils import DataSplit

SEED = 2333

def main():
    plan = pd.read_excel(PROCESSED_DIR / 'plan.xlsx', sheet_name='merge', dtype={MBDataKey.NUMBER: 'string'})
    plan.set_index(MBDataKey.NUMBER, inplace=True)
    child_idx = plan['group'] == MBGroup.CHILD
    child_plan = plan[child_idx]
    split_plan = pd.DataFrame(index=plan.index)
    split_plan.loc[~child_idx, 'split'] = DataSplit.TRAIN
    child_cases, child_subgroups = child_plan.index, child_plan['subgroup']
    gen_test_seed = 1
    while True:
        print(gen_test_seed)
        child_fit_cases, child_test_cases, child_fit_subgroups, _ = train_test_split(
            child_cases, child_subgroups,
            test_size=1 / 3,
            random_state=gen_test_seed,
            shuffle=True,
            stratify=child_subgroups,
        )
        # 强行归到测试集
        if '586779' in child_test_cases:
            break
        gen_test_seed += 1
    split_plan.loc[child_test_cases, 'split'] = DataSplit.TEST

    cv_parts = partition_dataset_classes(
        child_fit_cases,
        child_fit_subgroups,
        num_partitions=5,
        shuffle=True,
        seed=SEED,
    )
    for i, part in enumerate(cv_parts):
        for case in part:
            split_plan.at[case, 'split'] = i

    split_plan.to_excel(PROCESSED_DIR / 'split.xlsx', freeze_panes=(1, 0))

if __name__ == '__main__':
    main()
