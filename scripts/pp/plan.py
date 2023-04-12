import itertools
from pathlib import Path

import numpy as np
from pandas import ExcelWriter
import pandas as pd
import nibabel as nib
from tqdm import tqdm

from mbs.datamodule import DATA_DIR
from mbs.utils.enums import MBGroup

SUBGROUP_KEY = 'Subgroup'

def read_subgroup_tables():
    subgroup_tables = pd.read_excel(DATA_DIR / 'subgroup.xlsx', sheet_name=['儿童', '成人'])
    for group, rename in [
        ('儿童', MBGroup.CHILD),
        ('成人', MBGroup.ADULT),
    ]:
        subgroup_tables[rename] = subgroup_tables.pop(group)

    subgroup_tables[MBGroup.CHILD].rename(
        columns={
            '住院号': 'No.',
            '分子亚型': SUBGROUP_KEY,
        },
        inplace=True,
    )
    for table in subgroup_tables.values():
        table.set_index('No.', inplace=True)
        table.index = table.index.astype(str)
    return subgroup_tables

subgroup_tables = read_subgroup_tables()

mismatch_subgroup = []

def process_patient(patient_dir: Path):
    group = patient_dir.parent.name
    cur_table = subgroup_tables[group]
    patient = patient_dir.name
    patient_num = patient[:6]

    affine = None
    shape = None
    not_close = False
    exclude = False
    note = ''
    # T2 is at first
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
            affine_t = img.affine
            for i in range(3):
                affine_t[:, i] *= img.shape[i] / shape[i]
            if not np.allclose(affine_t, affine, atol=1e-2, rtol=1e-2):
                not_close = True
            if img.shape[2] != shape[2]:
                note = f'{img_type} slice number mismatch'
                exclude = True
    if not_close:
        note = 'affine not close'

    if patient_num not in cur_table.index:
        subgroup = ''
        note = 'subgroup not found'
        exclude = True
    else:
        subgroup = cur_table.loc[patient_num, SUBGROUP_KEY]
        if isinstance(subgroup, pd.Series):
            for i in range(1, len(subgroup)):
                if subgroup[i] != subgroup[0]:
                    exclude = True
                    note = 'subgroup inconsistent'
                    mismatch_subgroup.append(patient)

            subgroup = subgroup[0]

    if group == MBGroup.CHILD and patient_num in (adult_table := subgroup_tables[MBGroup.ADULT]).index:
        if subgroup == adult_table.loc[patient_num, SUBGROUP_KEY]:
            note = 'duplicate'
        else:
            exclude = True
            note = 'duplicate & inconsistent subgroup'
            mismatch_subgroup.append(patient)

    if group == MBGroup.ADULT and patient_num in subgroup_tables[MBGroup.CHILD].index:
        exclude = True
        note = 'duplicate in children group'

    return {
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
        'note': note,
    }

def main():
    with ExcelWriter(DATA_DIR / 'plan.xlsx') as writer:
        data = [process_patient(patient_dir) for patient_dir in tqdm(list((DATA_DIR / 'image').iterdir()))]
        pd.DataFrame.from_records(data).to_excel(writer, sheet_name=MBGroup.CHILD, index=False)
        adult_data = [process_patient(patient_dir) for patient_dir in tqdm(list((DATA_DIR / 'adult').iterdir()))]
        pd.DataFrame.from_records(adult_data).to_excel(writer, sheet_name=MBGroup.ADULT, index=False)
        merge_data = list(filter(
            lambda x: not x['exclude'],
            itertools.chain(data, adult_data),
        ))
        pd.DataFrame.from_records(merge_data).drop(columns='exclude').to_excel(writer, sheet_name='merge', index=False)

if __name__ == '__main__':
    main()
