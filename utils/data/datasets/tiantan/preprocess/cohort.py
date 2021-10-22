import itertools
import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from monai.transforms import *
from tqdm.contrib.concurrent import process_map

from convert import output_dir as input_dir
from utils.dicom_utils import ScanProtocol

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

def parse_desc(desc: str) -> Optional[ScanProtocol]:
    if desc in desc_reject_list:
        return None
    desc = desc.lower()
    if 't2' in desc:
        return ScanProtocol.T2
    elif 't1' in desc:
        if '+c' in desc:
            return ScanProtocol.T1c
        else:
            return ScanProtocol.T1

    return None

def patient_select(row):
    _, (patient, sex, age, weight, subgroup) = row
    # group scans by shape
    # grouped_scans = {}
    # all_scans = []
    scan_names = sorted(map(lambda scan_path: scan_path.name[:-7], (input_dir / patient).glob('*.nii.gz')))
    if any(protocol.name not in scan_names for protocol in ScanProtocol):
        protocol_scans = {
            ScanProtocol.T1: [],
            ScanProtocol.T1c: None,
            ScanProtocol.T2: None,
        }

        def get_scan_path(scan_name):
            return input_dir / patient / f'{scan_name}.nii.gz'

        def get_info_path(scan_name):
            return input_dir / patient / f'{scan_name}.json'

        for scan_name in scan_names:
            data: np.ndarray = nib.load(get_scan_path(scan_name)).get_fdata()
            info_path = get_info_path(scan_name)
            try:
                info = json.load(open(info_path))
            except Exception as e:
                print('check output of dcm2niix', info_path)
                raise e

            desc: str = info['SeriesDescription']
            if data.ndim == 3 and 22 <= data.shape[2] <= 26:
                protocol = parse_desc(desc)
                if protocol is not None:
                    protocol_scans[protocol] = scan_name

        if protocol_scans[ScanProtocol.T2] is None or len(protocol_scans[ScanProtocol.T1]) == 0:
            print('\nt2', patient, subgroup)
            return []
        if protocol_scans[ScanProtocol.T1c] is None:
            if len(protocol_scans[ScanProtocol.T1]) == 1:
                print('\nt1c', patient, subgroup)
                return []
            protocol_scans[ScanProtocol.T1c] = protocol_scans[ScanProtocol.T1].pop()

        protocol_scans[ScanProtocol.T1] = protocol_scans[ScanProtocol.T1][0]
        for protocol, scan_name in protocol_scans.items():
            save_dir = input_dir / patient
            get_scan_path(scan_name).rename(save_dir / f'{protocol.name}.nii.gz')
            get_info_path(scan_name).rename(save_dir / f'{protocol.name}.json')

    return [{
        'patient': patient,
        'subgroup': subgroup,
        'sex': sex,
        'age': age,
        'weight': weight,
    }]

if __name__ == '__main__':
    patient_info = pd.read_csv(input_dir / 'patient_info.csv')
    process_map(check_las, list(Path(input_dir).glob('*/*.nii.gz')), ncols=80)
    cohort = itertools.chain(*process_map(patient_select, list(patient_info.iterrows()), ncols=80))
    json.dump(list(cohort), open('cohort.json', 'w'), indent=4, ensure_ascii=False)
