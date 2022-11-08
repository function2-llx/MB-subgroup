import pandas as pd

from mbs.datamodule import load_cohort
from mbs.utils.enums import MBDataKey, Modality, SUBGROUPS, SegClass

name_mapping = {
    'WNT': 'wnt',
    'SHH': 'shh',
    'G3': 'group3',
    'G4': 'group4',
}

def main():
    cohort = load_cohort()
    inputs = []
    for split, cases in cohort.items():
        for case in cases:
            for modality, seg_class in [
                (Modality.T1, SegClass.CT),
                (Modality.T2, SegClass.AT),
            ]:
                inputs.append({
                    'Image': str(case[modality]),
                    'Mask': str(case[seg_class]),
                    MBDataKey.CASE: case[MBDataKey.CASE],
                    'molecular':  name_mapping[SUBGROUPS[case[MBDataKey.SUBGROUP_ID]]],
                    'modality': str(modality).lower(),
                    'split': split,
                })

    pd.DataFrame.from_records(inputs).to_csv('extractive.csv', index=False)

if __name__ == '__main__':
    main()
