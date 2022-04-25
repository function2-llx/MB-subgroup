import itertools
import os
import random
from glob import glob, iglob
from pathlib import Path

import pandas as pd
import pydicom
from tqdm.contrib.concurrent import process_map

from mbs.utils.dicom_utils import Plane, get_plane

dcm_dir = Path('dcm')
output_dir = Path('nifti')

def process_patient(patient):
    # patient directory only contains one folder, which contains all the scans
    patient_dir = dcm_dir / patient
    patient_dir = next(patient_dir.iterdir())
    try:
        sample_slice_path = next(iglob(os.path.join(patient_dir, '**/*.dcm'), recursive=True))
    except StopIteration:
        return []

    sample_ds = pydicom.dcmread(sample_slice_path)
    sex = {
        'M': 'male',
        'F': 'female'
    }[sample_ds.PatientSex]
    age = int(sample_ds.PatientAge[:3])
    weight = float(sample_ds.PatientWeight)
    patient_output_dir = output_dir / patient
    patient_output_dir.mkdir(exist_ok=True)

    def process_scan(scan_dir):
        slices = glob(os.path.join(scan_dir, '*.dcm'))
        if len(slices) <= 6:
            return
        sample_ds = pydicom.dcmread(slices[0])
        plane = get_plane(sample_ds)
        if plane is not None and plane == Plane.Axial:
            # requires dcm2niix(https://github.com/rordenlab/dcm2niix) to be installed
            os.system(f'dcm2niix -z y -f %s -o {patient_output_dir} {scan_dir}')

    for scan in os.listdir(patient_dir):
        process_scan(os.path.join(patient_dir, scan))

    return [(patient, sex, age, weight, subgroup_dict[patient])]

if __name__ == '__main__':
    random.seed(233333)
    subgroup_dict = dict(pd.read_csv('subgroup.csv').values)
    descs = set()
    os.makedirs(output_dir, exist_ok=True)

    patients = []
    for patient in os.listdir(dcm_dir):
        if not (dcm_dir / patient).is_dir():
            continue
        patients.append(patient)

    patient_info = list(itertools.chain(*process_map(process_patient, patients, ncols=80)))
    pd.DataFrame(patient_info, columns=['patient', 'sex', 'age', 'weight', 'subgroup']) \
        .to_csv(output_dir / 'patient_info.csv', index=False)
