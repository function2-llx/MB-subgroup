import os
import random
from glob import glob, iglob
from pathlib import Path

import pandas as pd
import pydicom
from monai.data import ITKReader
from monai.transforms import LoadImage
from tqdm import tqdm

from utils.dicom_utils import Plane, get_plane

data_dir = 'origin'
output_dir = Path('nifti')

def process_patient(patient, patient_dir):
    try:
        sample_slice_path = next(iglob(os.path.join(patient_dir, '**/*.dcm'), recursive=True))
    except StopIteration:
        return

    sample_ds = pydicom.dcmread(sample_slice_path)
    sex = {
        'M': 'male',
        'F': 'female'
    }[sample_ds.PatientSex]
    age = int(sample_ds.PatientAge[:3])
    weight = float(sample_ds.PatientWeight)
    patient_info.append((patient, sex, age, weight, subgroup_dict[patient]))

    patient_output_dir = os.path.join(output_dir, patient)
    os.makedirs(os.path.join(output_dir, patient), exist_ok=True)

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

if __name__ == '__main__':
    random.seed(233333)
    subgroup_dict = dict(pd.read_csv('subgroup.csv').values)
    patient_info = []
    descs = set()
    scan_info = []
    os.makedirs(output_dir, exist_ok=True)
    loader = LoadImage(ITKReader())

    for patient in tqdm(os.listdir(data_dir), ncols=80):
        patient_dir = os.path.join(data_dir, patient)
        if not os.path.isdir(patient_dir):
            continue
        # patient directory only contains one folder, which contains all the scans
        patient_dir = os.path.join(patient_dir, os.listdir(patient_dir)[0])
        process_patient(patient, patient_dir)
        # pd.DataFrame(scan_info, columns=('dir', 'desc', 'n')).to_csv('descs.csv', index=False)

    pd.DataFrame(patient_info, columns=['patient', 'sex', 'age', 'weight', 'subgroup']) \
        .to_csv(os.path.join(output_dir, 'patient_info.csv'), index=False)
