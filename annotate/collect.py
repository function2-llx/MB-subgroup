# collect annotations

import csv
import os

data_dir = '../data'

if __name__ == '__main__':
    all_labels = []
    for patient in os.listdir(data_dir):
        patient_dir = os.path.join(data_dir, patient)
        if not os.path.isdir(patient_dir):
            continue
        label_path = os.path.join(patient_dir, 'label.csv')
        label = list(csv.reader(open(label_path)))
        print(patient)
        for path, exists in label:
            path = path[path.find(patient):].replace('\\', '/')
            exists = int(exists)
            if exists not in [0, 1, 2]:
                print(path)
                continue
            if exists in [0, 1]:
                exists = 1 - exists
            all_labels.append((path, exists))
    csv.writer(open(os.path.join(data_dir, 'exists.csv'), 'w')).writerows(all_labels)
