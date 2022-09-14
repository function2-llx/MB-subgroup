from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

from mbs.datamodule import DATA_DIR

subgroup = pd.read_excel(DATA_DIR / 'subgroup.xlsx', sheet_name='儿童')
subgroup.set_index('姓名', inplace=True)

def process_patient(patient_dir: Path):
    patient = patient_dir.name

    affine = None
    shape = None
    not_close = False
    exclude = False
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
                # print(patient, img_type)
                # print(img.shape)
                # print(affine_t)
            if img.shape[2] != shape[2]:
                print('slices no')
                exclude = True
    if not_close:
        print(patient, 'T2')
        print(affine)
        print(shape)

    if patient not in subgroup.index:
        sg = ''
        exclude = True
    else:
        sg = subgroup.loc[patient, '分子亚型']

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
        'subgroup': sg,
    }

def main():
    data = [process_patient(patient_dir) for patient_dir in tqdm(list((DATA_DIR / 'image').iterdir()))]
    pd.DataFrame.from_records(data).to_excel(DATA_DIR / 'plan.xlsx', index=False)


if __name__ == '__main__':
    main()
