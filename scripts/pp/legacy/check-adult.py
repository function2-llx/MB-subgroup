from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
import nibabel as nib

from mbs.datamodule import DATA_DIR
from mbs.utils.enums import Modality, SegClass

subgroup = pd.read_excel(DATA_DIR / 'subgroup.xlsx', sheet_name='成人', dtype=str).set_index('No.')

def check_name():
    for patient_dir in (DATA_DIR / 'adult').iterdir():
        patient = patient_dir.name
        patient_num = patient[:6]
        if patient_num not in subgroup.index:
            print(patient_dir.name, 'no sg')
        for img_type in list(Modality) + list(SegClass):
            img_path = patient_dir / f'{img_type}.nii'
            if not img_path.exists():
                for suf in ['.nii.gz', '.nia']:
                    if (alt := img_path.with_suffix(suf)).exists():
                        print(patient, alt.name)
                        break
                else:
                    print(patient_dir.name, img_type)

def check_seg(patient_dir: Path):
    for seg_class in SegClass:
        seg: nib.Nifti1Image = nib.load(patient_dir / f'{seg_class}.nii')
        v = np.unique(seg.get_fdata())
        if not np.array_equiv(v, [0, 1]):
            print(patient_dir.name)

def main():
    check_name()
    process_map(check_seg, list((DATA_DIR / 'adult').iterdir()))

if __name__ == '__main__':
    main()
