from pathlib import Path
from collections import OrderedDict
import re
import json
import itertools

import numpy as np
import nibabel as nib
from monai.transforms import *
from tqdm.contrib.concurrent import process_map
import pandas as pd

from convert import output_dir as input_dir

codes = 'LAS'

loader = Compose([
    LoadImageD('img'),
    AddChannelD('img'),
])

orientation = OrientationD('img', codes)

def check_las(filepath):
    img: nib.Nifti1Image = nib.load(filepath)
    # data: np.ndarray = img.get_fdata()
    if ''.join(nib.orientations.aff2axcodes(img.affine)) != 'LAS':
        print(filepath, img.get_fdata().shape)

patient_info = pd.read_csv(input_dir / 'patient_info.csv')
desc_reject_list = [
    'Apparent Diffusion Coefficient (mm2/s)',
    'Ax DWI 1000b',
    'Calibration',
    'diffusion',
    'ep2d_diff_3scan_trace_p2',
    'ep2d_diffusion',
    'Exponential Apparent Diffusion Coefficient',
    'FSE100-9',
    'I_t1_tse_sag_384',
    'OAx DWI Asset',
    'PU:OAx DWI Asset',
    'SE12_HiCNR',
    'TOF_3D_multi-slab',
]

def patient_select(row):
    _, (patient, sex, age, weight, subgroup) = row
    # group scans by shape
    grouped_scans = {}
    all_scans = []
    raw_all_scans = sorted(
        (input_dir / patient).glob('*.nii.gz'),
        key=lambda x: x.name[:-7],
    )

    for scan in raw_all_scans:
        data: np.ndarray = nib.load(scan).get_fdata()
        info = json.load(open(re.sub(r'\.nii\.gz$', '.json', str(scan))))
        desc = info['SeriesDescription']
        if data.ndim == 3 and 22 <= data.shape[2] <= 26 and desc not in desc_reject_list:
            grouped_scans.setdefault(data.shape, []).append(scan)
            all_scans.append(scan)

    for scans in grouped_scans.values():
        if len(scans) >= 3:
            scans = scans[:3]
            break
    else:
        if len(all_scans) >= 3:
            scans = all_scans[:3]
        else:
            print(patient)
            return []

    scans = sorted(scans, key=lambda scan: json.load(open(re.sub(r'\.nii\.gz$', '.json', str(scan))))['EchoTime'])
    scans = list(map(lambda x: x.name[:-7], scans))
    return [{
        'patient': patient,
        'subgroup': subgroup,
        'sex': sex,
        'age': age,
        'weight': weight,
        'scans': scans,
    }]

if __name__ == '__main__':
    print(input_dir)
    process_map(check_las, list(Path(input_dir).glob('*/*.nii.gz')), ncols=80)
    cohort = itertools.chain(*process_map(patient_select, list(patient_info.iterrows()), ncols=80))
    json.dump(list(cohort), open('cohort.json', 'w'), indent=4, ensure_ascii=False)
