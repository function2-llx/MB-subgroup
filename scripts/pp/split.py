import pandas as pd
from sklearn.model_selection import train_test_split

from mbs.datamodule import DATA_DIR
from mbs.utils.enums import MBGroup
from monai.data import partition_dataset_classes
from umei.utils import DataSplit

SEED = 2333

def main():
    cohort = pd.read_excel(DATA_DIR / 'plan.xlsx', sheet_name='merge').set_index('name')
    children_idx = cohort['group'] == MBGroup.CHILDREN
    children_cohort = cohort[children_idx]
    cohort.loc[~children_idx, 'split'] = DataSplit.TRAIN
    names, subgroups = children_cohort.index, children_cohort['subgroup']
    fit_names, test_names, subgroups, _ = train_test_split(
        names, subgroups,
        test_size=1 / 3,
        random_state=SEED,
        shuffle=True,
        stratify=subgroups,
    )
    for name in test_names:
        cohort.loc[name, 'split'] = DataSplit.TEST


    cv_parts = partition_dataset_classes(
        fit_names,
        subgroups,
        num_partitions=5,
        shuffle=True,
        seed=SEED,
    )
    for i, part in enumerate(cv_parts):
        for name in part:
            cohort.loc[name, 'split'] = i

    cohort.to_excel(DATA_DIR / 'plan-split.xlsx')

if __name__ == '__main__':
    main()
