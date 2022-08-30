from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

data_dir = Path('origin')
cohort = pd.read_excel(data_dir / 'patients.xlsx', sheet_name='儿童')
cohort.set_index('姓名', inplace=True)

def process_patient(patient_dir: Path):
    patient = patient_dir.name
    if patient not in cohort.index:
        print(patient)

    affine = None
    shape = None
    not_close = False
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
                print(patient, img_type)
                print(img.shape)
                print(affine_t)
            if img.shape[2] != shape[2]:
                print('???')
    if not_close:
        print(patient, 'T2')
        print(affine)
        print(shape)
    return {
        'name': patient,
        **{
            f'p{i}': np.linalg.norm(affine[:, i])
            for i in range(3)
        },
        **{
            f's{i}': shape[i]
            for i in range(3)
        }
    }

def main():
    data = [process_patient(patient_dir) for patient_dir in (data_dir / 'image').iterdir()]
    pd.DataFrame.from_records(data).to_excel('plan.xlsx', index=False)

if __name__ == '__main__':
    main()
