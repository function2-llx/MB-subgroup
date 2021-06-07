import json
import operator

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate

from utils.data.datasets.tiantan import dataset_dir

n_folds = 3

if __name__ == '__main__':
    features = np.load('features.npz')
    cohort = {
        info['patient']: info
        for info in json.load(open(dataset_dir / 'cohort.json'))
    }
    folds = json.load(open(dataset_dir / f'folds-{n_folds}.json'))
    targets = ['WNT', 'SHH', 'G3', 'G4']
    X = []
    y = []
    np.random.seed(2333)

    for val_id in range(n_folds):
        X_train, y_train, X_val, y_val = [], [], [], []
        for fold_id, fold in enumerate(folds):
            X = fold_id
            for patient in fold:



    for info in cohort:
        X.append(features[patient])
        y.append(subgroup)

    svm = LinearSVC(random_state=2333)
    results = cross_validate(svm, X, y, cv=3)
    print(results)
