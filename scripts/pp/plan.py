import itertools
from pathlib import Path

import numpy as np
from pandas import ExcelWriter
import pandas as pd
import nibabel as nib
from tqdm import tqdm

from mbs.datamodule import DATA_DIR, PROCESSED_DIR
from mbs.utils.enums import MBDataKey, MBGroup

subgroup_tables = pd.read_excel(PROCESSED_DIR / 'subgroup.xlsx', sheet_name=list(MBGroup), dtype={'case': 'string'})

mismatch_subgroup = []

def process_patient(patient_dir: Path):
    group = patient_dir.parent.name
    cur_table = subgroup_tables[group].set_index('case')
    patient = patient_dir.name
    patient_num = patient[:6]

    affine = None
    shape = None
    not_close = False
    exclude = False
    notes = []
    # T2 is at firstm
    for img_type in ['T2', 'T1', 'T1C', 'AT', 'CT', 'ST']:
        img_path = patient_dir / f'{img_type}.nii'
        if not img_path.exists():
            print(patient, img_type)
            continue
        img: nib.Nifti1Image = nib.load(img_path)
        if affine is None:
            affine = img.affine
            shape = img.shape
        else:
            if not np.allclose(img.affine, affine, atol=1e-2, rtol=1e-2):
                not_close = True
            if img.shape[2] != shape[2]:
                notes.append(f'{img_type} slice number mismatch')
                exclude = True
    if not_close:
        notes.append('affine not close')

    if patient_num not in cur_table.index:
        subgroup = ''
        notes.append('subgroup not found')
        exclude = True
    else:
        subgroup = cur_table.at[patient_num, MBDataKey.SUBGROUP]

    return {
        'number': patient_num,
        'name': patient,
        **{
            f'p{i}': np.linalg.norm(affine[:, i])
            for i in range(3)
        },
        **{
            f's{i}': shape[i]
            for i in range(3)
        },
        'exclude': exclude,
        'subgroup': subgroup,
        'group': group,
        'note': '\n'.join(notes),
    }

def main():
    with ExcelWriter(PROCESSED_DIR / 'plan.xlsx') as writer:
        child_data = [process_patient(patient_dir) for patient_dir in tqdm(list((DATA_DIR / 'child').iterdir()))]
        pd.DataFrame(child_data).to_excel(writer, sheet_name=MBGroup.CHILD, index=False, freeze_panes=(1, 0))
        adult_data = [process_patient(patient_dir) for patient_dir in tqdm(list((DATA_DIR / 'adult').iterdir()))]
        pd.DataFrame(adult_data).to_excel(writer, sheet_name=MBGroup.ADULT, index=False, freeze_panes=(1, 0))
        merge_data = list(filter(
            lambda x: not x['exclude'],
            itertools.chain(child_data, adult_data),
        ))
        pd.DataFrame(merge_data).drop(columns='exclude').to_excel(writer, sheet_name='merge', index=False, freeze_panes=(1, 0))

if __name__ == '__main__':
    main()
