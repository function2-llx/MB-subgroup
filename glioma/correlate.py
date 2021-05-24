import numpy as np
import pandas as pd

from FisherExact import fisher_exact

if __name__ == '__main__':
    subtypes = pd.read_excel('subtypes.xlsx')[['patient', 'IDH.codel.subtype']].set_index('patient')
    features = pd.read_csv('mri/features.csv').set_index('ID').drop('Date', axis=1)
    patients = subtypes.index.intersection(features.index)
    for patient in subtypes.index:
        if patient not in patients:
            subtypes.drop(patient, inplace=True)
    for patient in features.index:
        if patient not in patients:
            features.drop(patient, inplace=True)
    print('patients:', len(patients))
    # remove columns that has nan
    cnt = 0
    for feature_cls in features.columns:
        if features[feature_cls].isna().any() or (features[feature_cls] == '#DIV/0!').any() or (features[feature_cls] == np.inf).any():
            features.drop(feature_cls, axis=1, inplace=True)
            # print(feature_cls)
            cnt += 1
    print('has nan/invalid: ', cnt)
    print('rest:', len(features.columns))

    # normalize features to {0, 1, 2, 3}
    feature_split = 4
    for feature_cls, feature_col in features.iteritems():
        l = min(feature_col)
        r = max(feature_col)
        for patient, feature in feature_col.iteritems():
            for i in range(feature_split):
                th = l + (r - l) / feature_split * (i + 1)
                if feature <= th:
                    features.loc[patient, feature_cls] = i
                    break
            else:
                features.loc[patient, feature_cls] = feature_split - 1
    features = features.astype('int32')
    print(features)

    for subtype_cls in subtypes.columns:
        X = features.copy()
        y = subtypes[subtype_cls]
        for patient in patients:
            if pd.isna(y[patient]):
                X.drop(patient, inplace=True)
                y.drop(patient, inplace=True)
        types = set(y)
        print(subtype_cls, types)
        # map type to integer
        type_map = {
            t: i
            for i, t in enumerate(types)
        }
        y = [type_map[t] for t in y]
        feature_cnt = 0
        for feature_cls, feature_col in X.iteritems():
            table = np.zeros((feature_split, len(types)))
            for feature, subtype in zip(feature_col, y):
                table[feature, subtype] += 1
            print(table)
            p_value = fisher_exact(table)
            if p_value < 0.05 / len(X):
                print(feature_cls, subtype_cls, p_value)
                feature_cnt += 1
        print(feature_cnt)
        # clf = svm.SVC(kernel='linear', random_state=914)
        # results = cross_validate(clf, X, y, cv=5, verbose=10, n_jobs=15)
        # print(json.dumps(results, indent=4, ensure_ascii=False))
