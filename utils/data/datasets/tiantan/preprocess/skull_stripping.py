from sys import argv
from copy import deepcopy
import json
from pathlib import Path

from convert import output_dir as input_dir
from utils.dicom_utils import ScanProtocol


output_dir = Path('skull-stripped')

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
