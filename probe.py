import json

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import pandas as pd

from encode import output_dir
from utils.data.datasets.tiantan import dataset_dir

n_folds = 3

if __name__ == '__main__':
    from finetune import parse_args
    args = parse_args()
    features = np.load(output_dir / f'{args.features}.npz')
    cohort = json.load(open(dataset_dir / 'cohort.json'))
    # cohort = {
    #     info['patient']: info
    #     for info in json.load(open(dataset_dir / 'cohort.json'))
    # }
    # folds = json.load(open(dataset_dir / f'folds-{n_folds}.json'))
    targets = ['WNT', 'SHH', 'G3', 'G4']
    y = []
    patients = []
    # np.random.seed(914)
    for info in cohort:
        patients.append(info['patient'])
        y.append(info['subgroup'])

    table = []
    for features in ['', '-KM', '-B4']:
        model = f'ResNet-34{features}'
        features = np.load(output_dir / f'{model}.npz')
        X = list(map(lambda patient: features[patient], patients))
        svm = LinearSVC(random_state=914)
        results = cross_val_score(svm, X, y, cv=5)
        row = [model]
        row.extend(results.tolist())
        row.append(results.mean())
        table.append(row)
    df = pd.DataFrame(table, columns=['模型', '1折', '2折', '3折', '4折', '5折', '平均'])
    df.set_index('模型', inplace=True)
    print(df)
    df.to_csv('probe.csv', sep='\t')

    # for val_id in range(n_folds):
    #     X_train, y_train, X_val, y_val = [], [], [], []
    #     for fold_id, fold in enumerate(folds):
    #         X = fold_id
    #         for patient in fold:
    #
