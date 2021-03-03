import csv
import json
import os
import random
from random import sample

import itk
import numpy as np
import pydicom
from tqdm import trange
from monai.transforms import LoadImage

from utils.data import get_plane
from utils.enums import Plane

random.seed(233333)

data_dir = 'data-dicom'
subgroups = {
    1: 'WNT',
    2: 'SHH',
    3: 'G3',
    4: 'G4',
}

descs = set()

def process_scan(scan_dir: str, loader: LoadImage):
    # print(scan_dir)
    try:
        data, meta = loader(scan_dir)
    except:
        return

    # get plane of the whole scan
    slices = list(filter(lambda x: x.endswith('.dcm'),  os.listdir(scan_dir)))
    if len(slices) > 0:
        
    for slice in os.listdir(scan_dir):
        if not slice.endswith('.dcm'):
            continue
        ds = pydicom.dcmread(os.path.join(scan_dir, slice))
        print(ds.ManufacturerModelName)
        # exit(1)
        descs.add(ds.SeriesDescription)
        # print(ds.SeriesDescription)
        plane = get_plane(ds)
        break

    # print(plane)
    if plane is None or plane != Plane.Axial:
        return
    print(ds.SeriesDescription)


    # data = np.asarray(data)
    # # data = data.transpose(1, 2, 0)
    # print(data.shape)
    # data = itk.image_view_from_array(data)
    # save_name = f'{os.path.split(scan_dir)[1]}.nii.gz'
    # print(save_name)
    # itk.imwrite(data, save_name)

def process_patient(patient, patient_dir, loader):
    # patient directory only contains one folder, which contains all the scans
    patient_dir = os.path.join(patient_dir, os.listdir(patient_dir)[0])
    for scan in os.listdir(patient_dir):
        scan_dir = os.path.join(patient_dir, scan)
        # get patient sex, age
        for slice in os.listdir(scan_dir):
            if not slice.endswith('.dcm'):
                continue
            ds = pydicom.dcmread(os.path.join(scan_dir, slice))
            sex = ds.PatientSex
            age = ds.PatientAge
            weight = ds.PatientWeight

        process_scan(scan_dir, loader)

def process_3d():
    from monai.data import ITKReader
    loader = LoadImage(ITKReader())

    for patient in os.listdir(data_dir):
        patient_dir = os.path.join(data_dir, patient)
        if not os.path.isdir(patient_dir):
            continue
        process_patient(patient, patient_dir, loader)


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
    process_3d()
