import csv
from glob import glob
import json
import os
import random

import pydicom
from tqdm import trange

from utils import get_ornt

random.seed(233333)

data_dir = 'data'
subtypes = {
    1: 'WNT',
    2: 'SHH',
    3: 'G3',
    4: 'G4',
}

if __name__ == '__main__':
    subtype_table = dict(csv.reader(open(os.path.join(data_dir, 'tag.csv'))))
    for k, v in subtype_table.items():
        subtype_table[k] = int(v)
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
    bar = trange(tot, ncols=80)
    for patient in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient)
        if not os.path.isdir(patient_path):
            continue
        if patient not in subtype_table:
            print('lack subtype:', patient)
            continue
        split = random.choices(['train', 'val'], [3, 1])[0]
        subtype_idx = subtype_table[patient]
        patient_data = {
            'subtype_idx': subtype_idx,
            'subtype': subtypes[subtype_idx],
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
