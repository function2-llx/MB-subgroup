from dataclasses import dataclass
import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
from xgboost import XGBClassifier

@dataclass
class ModelConf:
    name: str
    model_cls: type
    init_kwargs: dict
    grid_params: dict[str, list]

model_registry = [
    # ModelConf('SVM', SVC, {'random_state': 42}, {
    #     'C': [0.1, 1, 10, 100, 1000],
    #     'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #     'kernel': ['rbf', 'sigmoid', 'linear'],
    # }),
    # ModelConf('LR', LogisticRegression, {'solver': 'saga', 'l1_ratio': 0.5, 'max_iter': 1000, 'random_state': 42}, {
    #     'penalty': ['l1', 'l2', 'elasticnet', None],
    #     'C': [10, 1, 0.1, 0.01, 0.001],
    # }),
    # ModelConf('KNN', KNeighborsClassifier, {}, {'n_neighbors': [3, 5, 7, 9]}),
    # ModelConf('RF', RandomForestClassifier, {'random_state': 42}, {
    #     'n_estimators': [50, 100, 200, 300],
    #     'max_depth': [1, 2, 3, 4],
    # }),
    # ModelConf('XGB', XGBClassifier, {'random_state': 42}, {
    #     'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    #     'max_depth': [3, 4, 5, 6],
    # }),
    # ModelConf('MLP', MLPClassifier, {'max_iter': 2000, 'random_state': 42}, {
    #     'hidden_layer_sizes': [(100, 100, 50), (50, 100, 50), (100, 50, 100)],
    #     'learning_rate': ['constant', 'invscaling', 'adaptive'],
    # }),
    ModelConf(
        'MLP',
        MLPClassifier, {
            'max_iter': 2000, 'random_state': 42,
            'hidden_layer_sizes': (100, 100, 50),
            'learning_rate': 'constant',
        },
        {},
    ),
]

model_names = [model_conf.name for model_conf in model_registry]

conf = OmegaConf.load(Path('conf.yaml'))
df = pd.read_csv('features.csv', index_col='case')
df['sex'] = df['sex'].map(lambda sex: {'F': 0, 'M': 1}.get(sex, -1))
df['age'] = df['age'].map(lambda x: 0 if pd.isna(x) else x)
test_idx = df['split'] == 'test'
train_idx = ~test_idx

def upsample_minority(X_train: pd.DataFrame, y_train: pd.Series):
    train = pd.concat([X_train, y_train], axis='columns')
    upsample_cls = (y_train == 0).sum() * 2 > len(train.index)
    upsample_idx: pd.Series = y_train == upsample_cls
    others_idx = ~upsample_idx
    upsampled = resample(train[upsample_idx], replace=True, n_samples=others_idx.sum(), random_state=42)
    upsampled_train = pd.concat([upsampled, train[others_idx]])
    upsampled_train = upsampled_train.sample(frac=1, random_state=42)
    upsampled_X_train: pd.DataFrame = upsampled_train.iloc[:, :-1]
    upsampled_y_train: pd.Series = upsampled_train.iloc[:, -1]
    return upsampled_X_train, upsampled_y_train

def standardize_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_train = pd.DataFrame(scaled_X_train, index=X_train.index, columns=X_train.columns)
    scaled_X_test = scaler.transform(X_test)
    scaled_X_test = pd.DataFrame(scaled_X_test, index=X_test.index, columns=X_test.columns)
    return scaled_X_train, scaled_X_test

def run_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, model_conf: ModelConf):
    init_kwargs = model_conf.init_kwargs
    model = model_conf.model_cls(**init_kwargs)
    grid = GridSearchCV(model, model_conf.grid_params, scoring='accuracy', cv=5, n_jobs=-1)
    print('grid search:', model_conf.name, model_conf.grid_params)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    if issubclass(model_conf.model_cls, SVC):
        init_kwargs = {'probability': True, **model_conf.init_kwargs}
    model = model_conf.model_cls(**init_kwargs, **grid.best_params_)
    bagging = BaggingClassifier(model, 5, random_state=42)
    bagging.fit(X_train, y_train)
    y_prob = bagging.predict_proba(X_test)
    if y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
        return pd.Series(y_prob, X_test.index)
    else:
        return pd.DataFrame(y_prob, X_test.index, ['wnt', 'shh', 'group3', 'group4'])

def run(stage: str, group: str | None = None):
    task_conf = conf[stage]
    if group is not None:
        task_conf = task_conf[group]
    # prepare data
    target = df["molecular"].map(task_conf['map'])
    task_train_idx = (target != -1) & train_idx
    reduced_features_list = task_conf['reduced']
    reduced_features = df[reduced_features_list]
    X_train = reduced_features[task_train_idx]
    X_test = reduced_features[test_idx]
    y_train = target[task_train_idx]

    X_train, y_train = upsample_minority(X_train, y_train)
    X_train, X_test = standardize_features(X_train, X_test)
    if stage == 'e2e':
        ret = {}
    else:
        ret = pd.DataFrame(index=X_test.index)
    for model_conf in model_registry:
        X_train = X_train.reset_index(drop=True)
        ret[model_conf.name] = run_model(X_train, y_train, X_test, model_conf)
    return ret

def get_sheet_name(stage, group):
    sheet_name = stage
    if group is not None:
        sheet_name += f'-{group}'
    return sheet_name

def main():
    if not (save_path := Path('prob-results.xlsx')).exists():
        with pd.ExcelWriter(save_path) as writer:
            for stage, group in [
                ('s1', None),
                ('s2', 'ws'),
                ('s2', 'g34'),
            ]:
                run(stage, group).to_excel(writer, get_sheet_name(stage, group))
            for model_name, result in run('e2e').items():
                result.to_excel(writer, f'e2e-{model_name}')

    g34_probs = pd.read_excel(save_path, sheet_name='s1', index_col='case')
    shh_probs = pd.read_excel(save_path, sheet_name='s2-ws', index_col='case')
    g4_probs = pd.read_excel(save_path, sheet_name='s2-g34', index_col='case')
    test_label = df['molecular'][test_idx]
    s1_true = test_label.map({'wnt': 0., 'shh': 0., 'group3': 1., 'group4': 1.})

    # cp: conditional probability
    def merge_prob(s1: str | None, ws: str, g34: str, cp: bool = False):
        if s1 is None:
            g34_prob = s1_true.to_numpy()
        else:
            g34_prob = g34_probs[s1].to_numpy()
        ws_prob = 1 - g34_prob
        shh_prob = shh_probs[ws].to_numpy()
        wnt_prob = 1 - shh_prob
        g4_prob = g4_probs[g34].to_numpy()
        g3_prob = 1 - g4_prob
        if cp:
            y_prob = np.column_stack([ws_prob * wnt_prob, ws_prob * shh_prob, g34_prob * g3_prob, g34_prob * g4_prob])
        else:
            y_prob = np.where(
                g34_prob < 0.5,
                np.vstack([wnt_prob, shh_prob, np.zeros((2, len(test_label.index)))]),
                np.vstack([np.zeros((2, len(test_label.index))), g3_prob, g4_prob]),
            ).T
        return y_prob

    def cal_metrics(y_prob):
        # import torch really slows down multiprocessing
        import torch
        from torchmetrics import AUROC, Accuracy, Recall, Specificity, F1Score
        from luolib.metrics import MulticlassBinaryAccuracy
        metrics = {
            'f1': F1Score('multiclass', num_classes=4, average=None),
            'sen': Recall('multiclass', num_classes=4, average=None),
            'spe': Specificity('multiclass', num_classes=4, average=None),
            'b-acc': MulticlassBinaryAccuracy(num_classes=4, average=None),
            'auroc': AUROC('multiclass', num_classes=4, average=None),
            'acc': Accuracy('multiclass', num_classes=4),
        }
        y_test = test_label.map({
            'wnt': 0,
            'shh': 1,
            'group3': 2,
            'group4': 3,
        }).to_numpy()
        y_test = torch.tensor(y_test)
        y_prob = torch.tensor(y_prob)
        return {
            name: metric(y_prob, y_test).numpy()
            for name, metric in metrics.items()
        }

    reports = []
    for s1, ws, g34, cp in it.product(*it.repeat(model_names, 3), [False, True]):
        report = cal_metrics(merge_prob(s1, ws, g34, cp))
        report['model'] = f'{s1}+{ws}+{g34}' + ('(cp)' if cp else '')
        if s1 == ws == g34 and not cp:
            report['model'] = s1
            reports.append(report)
    for model_name in model_names:
        y_prob = pd.read_excel(save_path, sheet_name=f'e2e-{model_name}', index_col='case').to_numpy()
        report = cal_metrics(y_prob)
        report['model'] = f'e2e-{model_name}'
        # reports.append(report)
    reports = pd.DataFrame(reports).set_index('model')
    metrics = {
        name: np.stack(metric.to_numpy())
        for name, metric in reports.items()
    }
    acc = metrics.pop('acc')
    with pd.ExcelWriter('reports.xlsx') as writer:
        pd.DataFrame({
            **{name: metric.mean(axis=-1) for name, metric in metrics.items()},
            'acc': acc,
        }, reports.index).to_excel(writer, sheet_name='macro', freeze_panes=(1, 1))
        for i, subgroup in enumerate(['wnt', 'shh', 'group3', 'group4']):
            pd.DataFrame({name: metric[:, i] for name, metric in metrics.items()}, reports.index) \
                .to_excel(writer, sheet_name=subgroup, freeze_panes=(1, 1))

if __name__ == '__main__':
    main()
