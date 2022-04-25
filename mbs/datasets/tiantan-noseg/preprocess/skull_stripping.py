import json
from pathlib import Path
from sys import argv

import numpy as np
import nibabel as nib
from tqdm.contrib.concurrent import process_map

from convert import output_dir as input_dir
from mbs.utils.dicom_utils import ScanProtocol

output_dir = Path('stripped')

def check(info):
    patient = info['patient']
    for protocol in ScanProtocol:
        origin = nib.load(input_dir / patient / f'{protocol.name}.nii.gz').get_fdata()
        stripped = nib.load(output_dir / patient / f'{protocol.name}.nii.gz').get_fdata()
        mask = nib.load(output_dir / patient / f'{protocol.name}_mask.nii.gz').get_fdata().astype(bool)
        masked = origin * mask
        assert np.array_equal(stripped, masked)

if __name__ == '__main__':
    l, r = map(int, argv[1:])
    print('range:', l, r)
    cohort = json.load(open('cohort.json'))
    mri_fnames = []
    out_fnames = []
    for info in cohort:
        patient = info['patient']
        for protocol in ScanProtocol:
            test = input_dir / patient / f'{protocol.name}.nii.gz'
            mri_fnames.append(str(test))
            out_fnames.append(str(output_dir / patient / f'{protocol.name}.nii.gz'))

    from HD_BET.run import run_hd_bet
    run_hd_bet(mri_fnames[l:r], out_fnames[l:r], postprocess=True)
    process_map(check, cohort, ncols=80, desc='checking mask')
