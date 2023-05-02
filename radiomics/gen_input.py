from pathlib import Path

import pandas as pd

from mbs.datamodule import load_merged_plan, load_split
from mbs.utils.enums import MBDataKey, Modality, SUBGROUPS, SegClass

name_mapping = {
    'WNT': 'wnt',
    'SHH': 'shh',
    'G3': 'group3',
    'G4': 'group4',
}

data_dir = Path('../mbs/processed/cr-p10/register-crop')

def main():
    plan = load_merged_plan()
    split = load_split()
    inputs = []
    for case in plan.index:
        if not (mask_path := data_dir / case / f'{SegClass.AT}.nii').exists():
            mask_path = Path('pred-seg') / f'{case}.nii'
        for modality in [Modality.T1, Modality.T2]:
            inputs.append({
                'Image': str(data_dir / case / f'{modality}.nii'),
                'Mask': str(mask_path),
                'case': case,
                'molecular':  name_mapping[plan.at[case, MBDataKey.SUBGROUP]],
                'modality': str(modality).lower(),
                'split': split[case],
            })

    pd.DataFrame.from_records(inputs).to_csv('batch.csv', index=False)

if __name__ == '__main__':
    main()
