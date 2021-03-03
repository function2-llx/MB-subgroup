# data statistics

import os
from collections import Counter

from tqdm import tqdm
import pydicom

from utils.data import get_plane

data_dir = 'data-dicom'


def all_same(list):
    return all([x == list[0] for x in list])

def process_scan(scan_dir):
    slices = os.listdir(scan_dir)
    if len(slices) <= 3:
        return
    series_descriptions = []
    planes = []
    for slice in slices:
        path = os.path.join(scan_dir, slice)
        ds = pydicom.dcmread(path)
        series_descriptions.append(ds.SeriesDescription)
        try:
            planes.append(get_plane(ds)[0])
        except:
            print(path)
    if not all_same(series_descriptions) or not all_same(planes):
        print(scan_dir)
        print(series_descriptions, planes)

if __name__ == '__main__':
    n_patients = 0
    for patient in os.listdir(data_dir):
        patient_dir = os.path.join(data_dir, patient)
        if not os.path.isdir(patient_dir):
            continue
        n_patients += 1
        patient_dir = os.path.join(patient_dir, os.listdir(patient_dir)[0])
        for scan_dir in os.listdir(patient_dir):
            process_scan(os.path.join(patient_dir, scan_dir))

# stat = Counter()
# n_instances = 0

# for dirpath, dirnames, filenames in os.walk(data_dir):
#     for filename in filenames:
#         filepath = os.path.join(dirpath, filename)
#         if filepath.endswith('.dcm'):
#             n_instances += 1


# for dirpath, dirnames, filenames in tqdm(os.walk(data_dir), ncols=80, total=n_instances):
#     for filename in filenames:
#         filepath = os.path.join(dirpath, filename)
#         if not filepath.endswith('.dcm'):
#             # all_dcms.append(filepath)
#             continue
#         # print(filepath)
#         ds = pydicom.dcmread(filepath)
#         stat[ds.SeriesDescription] += 1
#         print(ds.SeriesDescription)

# print(stat)
