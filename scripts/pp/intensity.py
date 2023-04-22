import nibabel as nib
import pandas as pd
from tqdm.contrib.concurrent import process_map

from mbs.utils.enums import MBDataKey, Modality, DATA_DIR, PROCESSED_DIR

plan = pd.read_excel(PROCESSED_DIR / 'plan.xlsx', sheet_name='merge', dtype={MBDataKey.NUMBER: 'string'})
plan.set_index(MBDataKey.NUMBER, inplace=True)

def process(number: str):
    name = plan.at[number, 'name']
    group = plan.at[number, MBDataKey.GROUP]
    ret = {
        MBDataKey.NUMBER: number,
        MBDataKey.GROUP: group,
    }
    for modality in Modality:
        data = nib.load(DATA_DIR / group / name / f'{modality}.nii').get_fdata()
        ret[f'{modality}-min'] = data.min()
        ret[f'{modality}-max'] = data.max()
    return ret

def main():
    results = process_map(process, plan.index)
    pd.DataFrame(results).to_excel(PROCESSED_DIR / 'intensity.xlsx', index=False)

if __name__ == '__main__':
    main()
