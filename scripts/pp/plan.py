import itertools
from pathlib import Path

import numpy as np
from pandas import ExcelWriter
import pandas as pd
import nibabel as nib
from tqdm import tqdm

from mbs.utils.enums import MBDataKey, MBGroup, Modality, SegClass, DATA_DIR, PROCESSED_DIR, SEG_REF

subgroup_tables = pd.read_excel(PROCESSED_DIR / 'subgroup.xlsx', sheet_name=list(MBGroup), dtype={'case': 'string'})

mismatch_subgroup = []

def process_patient(patient_dir: Path):
    group = patient_dir.parent.name
    cur_table = subgroup_tables[group].set_index('case')
    patient = patient_dir.name
    patient_num = patient[:6]

    exclude = False
    notes = []
    affine = {}
    shape = {}
    for img_type in [*Modality, *SegClass]:
        img_path = patient_dir / f'{img_type}.nii'
        if not img_path.exists():
            print(patient, img_type)
            continue
        img: nib.Nifti1Image = nib.load(img_path)
        affine[img_type] = img.affine
        shape[img_type] = img.shape
    for modality in [Modality.T1, Modality.T1C]:
        if shape[modality][-1] != shape[Modality.T2][-1]:
            notes.append(f'{modality} slice number mismatch')

    for seg_class in SegClass:
        if seg_class not in affine:
            continue
        if (match_type := SEG_REF.get(seg_class, None)) is not None:
            if not np.allclose(affine[seg_class], affine[match_type], atol=1e-3, rtol=1e-3):
                notes.append(f'{seg_class} affine not close to {match_type}')
            if shape[seg_class] != shape[match_type]:
                notes.append(f'{seg_class} shape not equal to {match_type}')

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
            f'p{i}': np.linalg.norm(affine[Modality.T2][:, i])
            for i in range(3)
        },
        **{
            f's{i}': shape[Modality.T2][i]
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
