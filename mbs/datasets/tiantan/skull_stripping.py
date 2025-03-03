from pathlib import Path
from sys import argv

import nibabel as nib
import numpy as np
import pandas as pd

from mbs.utils.dicom_utils import ScanProtocol

input_dir = Path('origin')
output_dir = Path('stripped')

def check(info):
    patient, info = info
    for protocol in ScanProtocol:
        origin = nib.load(input_dir / patient / info[protocol.name]).get_fdata()
        stripped = nib.load(output_dir / patient / f'{protocol.name}.nii.gz').get_fdata()
        mask = nib.load(output_dir / patient / f'{protocol.name}_mask.nii.gz').get_fdata().astype(bool)
        masked = origin * mask
        assert np.array_equal(stripped, masked)

def main():
    cohort = pd.read_excel('cohort.xlsx', index_col='name(raw)')
    mri_fnames = []
    out_fnames = []
    for patient, info in cohort.iterrows():
        for protocol in ScanProtocol:
            input_fname = input_dir / patient / info[protocol.name]
            mri_fnames.append(str(input_fname))
            output_fname = output_dir / patient / info[protocol.name]
            output_fname.parent.mkdir(parents=True, exist_ok=True)
            out_fnames.append(str(output_fname))
    print('total:', len(mri_fnames))
    l, r = map(int, argv[1:])
    print('range:', l, r)
    from HD_BET.run import run_hd_bet
    run_hd_bet(mri_fnames[l:r], out_fnames[l:r])
    # process_map(check, cohort.iterrows(), ncols=80, desc='checking mask')

if __name__ == '__main__':
    main()
