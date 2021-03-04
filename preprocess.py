import csv
import json
import os
import random
from collections import Counter
from glob import glob, iglob
from random import sample
from typing import Optional
import sys

import itk
import numpy as np
import pandas as pd
import pydicom
from monai.data import ITKReader
from monai.transforms import LoadImage
from tqdm import tqdm
from tqdm import trange

from utils.dicom import parse_series_desc, Plane, get_plane

random.seed(233333)

data_dir = 'data-dicom'
output_dir = 'processed_3d'
subgroup_dict = dict(pd.read_csv('subgroup.csv').values)
patient_info = []

# subgroups = {
#     1: 'WNT',
#     2: 'SHH',
#     3: 'G3',
#     4: 'G4',
# }

descs = set()
scan_info = []

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

def process_2d():
    subgroup_table = dict(csv.reader(open(os.path.join(data_dir, 'tag.csv'))))
    for k, v in subgroup_table.items():
        subgroup_table[k] = int(v)
    exists_table = dict(csv.reader(open(os.path.join(data_dir, 'exists.csv'))))
    for k, v in exists_table.items():
        exists_table[k] = int(v)

    tot = 0
    for _, _, filenames in os.walk(data_dir):
        for filename in filenames:
            tot += filename.endswith('.png')

    splits = {
        split: {}
        for split in ['train', 'val']
    }
    all_patients = {}
    for patient in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient)
        if not os.path.isdir(patient_path):
            continue
        if patient not in subgroup_table:
            print('cannot find subgroup for', patient)
            continue
        split = random.choices(['train', 'val'], [3, 1])[0]
        subgroup_idx = subgroup_table[patient]
        all_patients.setdefault(subgroup_idx, []).append(patient)

    bar = trange(tot, ncols=80)
    for subgroup_idx, patients in all_patients.items():
        train_patients = set(sample(patients, int(len(patients) / 4 * 3)))
        for patient in patients:
            split = 'train' if patient in train_patients else 'val'
            patient_path = os.path.join(data_dir, patient)
            patient_data = {
                'subgroup_idx': subgroup_idx,
                'subgroup': subgroups[subgroup_idx],
            }
            for ortn in ['back', 'up', 'left']:
                patient_data[ortn] = []
            for dirpath, _, filenames in os.walk(patient_path):
                for filename in filenames:
                    if not filename.endswith('.png'):
                        continue
                    ds = pydicom.dcmread(os.path.join(dirpath, filename[:-4] + '.dcm'))
                    try:
                        ortn, ortn_id = get_ornt(ds)
                    except AttributeError:
                        bar.update()
                        continue
                    file_relpath = os.path.relpath(os.path.join(dirpath, filename), data_dir)
                    exists = exists_table[file_relpath]
                    if exists in [0, 1]:
                        patient_data[ortn].append({
                            'path': file_relpath,
                            'exists': bool(exists),
                        })
                    else:
                        os.remove(os.path.join(data_dir, file_relpath))
                        os.remove(os.path.join(data_dir, file_relpath[:-4] + '.dcm'))
                    bar.update()
            splits[split][patient] = patient_data

    bar.close()
    for split, data in splits.items():
        json.dump(data, open(os.path.join(data_dir, f'{split}.json'), 'w'), ensure_ascii=False, indent=4)

if __name__ == '__main__':
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
