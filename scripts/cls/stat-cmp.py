from enum import StrEnum
from luolib.types import PathLike

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from statsmodels.stats.contingency_tables import mcnemar

from mbs.utils.enums import SUBGROUPS

pROC = importr('pROC')

def read_case_outputs(filepath: PathLike):
    df = pd.read_excel(filepath, dtype={'case': 'string'}).set_index('case')
    return df

class Metric(StrEnum):
    SEN = 'sensitivity'
    SPE = 'specificity'
    ACC = 'accuracy'

def report_metric_map(metric: str):
    return {
        'sensitivity': 'sen',
        'specificity': 'spe',
        'accuracy': 'acc',
        'auc': 'auroc',
    }[metric]

def mcnemar_table(p1: np.ndarray, p2: np.ndarray):
    table = np.array([
        [(p1 & p2).sum(), (~p1 & p2).sum()],
        [(p1 & ~p2).sum(), (~p1 & ~p2).sum()],
    ])

    return table, table[0, 1] + table[1, 0] < 25

metrics_list = [
    'f1', 'sensitivity', 'specificity', 'accuracy', 'auc'
]

def get_cls_index(cls: str):
    if cls == 'macro':
        return len(SUBGROUPS)
    else:
        return SUBGROUPS.index(cls)

def compare_predictions(split: str, m1: str, m2: str) -> pd.DataFrame:
    o1 = read_case_outputs(f'MB-data/outputs/{split}/{m1}.xlsx')
    o2 = read_case_outputs(f'MB-data/outputs/{split}/{m2}.xlsx')
    assert ((true := o1['true'].to_numpy()) == o2['true'].to_numpy()).all()
    p1 = o1['pred'].to_numpy()
    p2 = o2['pred'].to_numpy()
    diff_bs = np.load(f'MB-data/metrics/{split}/{m1}/bs_metrics.npy') - np.load(f'MB-data/metrics/{split}/{m2}/bs_metrics.npy')
    ci = np.stack(
        [
            np.percentile(diff_bs, q=2.5, axis=0),
            np.percentile(diff_bs, q=97.5, axis=0),
        ],
        axis=-1,
    )
    report1 = pd.read_excel(f'MB-data/metrics/{split}/{m1}/report.xlsx', index_col=0)
    report2 = pd.read_excel(f'MB-data/metrics/{split}/{m2}/report.xlsx', index_col=0)

    results = []
    
    for cls in SUBGROUPS:
        cls_mask = true == cls
        p1b = (p1 == cls) == cls_mask
        p2b = (p2 == cls) == cls_mask
        
        # McNemar's test for different metrics
        for metric in Metric:
            match metric:
                case Metric.SEN:
                    sub_idx = cls_mask
                case Metric.SPE:
                    sub_idx = ~cls_mask
                case Metric.ACC:
                    sub_idx = slice(None)
                    
            table, exact = mcnemar_table(p1b[sub_idx], p2b[sub_idx])
            result = mcnemar(table, exact=False, correction=table[1, 0] == table[0, 1])
            p = result.pvalue
            results.append({
                'cls': cls,
                'metric': metric,
                'value': report2.loc[cls, report_metric_map(metric)],
                'diff': report1.loc[cls, report_metric_map(metric)] - report2.loc[cls, report_metric_map(metric)],
                'p': p,
            })
        
        # AUC comparison using DeLong's test
        true_r = robjects.FloatVector(cls_mask)
        prob1_r = robjects.FloatVector(o1[cls].to_numpy())
        prob2_r = robjects.FloatVector(o2[cls].to_numpy())
        roc_1 = pROC.roc(true_r, prob1_r)
        roc_2 = pROC.roc(true_r, prob2_r)
        print(cls, roc_1.rx2('auc')[0], roc_2.rx2('auc')[0])
        test_result = pROC.roc_test(roc_1, roc_2, method='delong')
        print('auc', p := test_result.rx2('p.value')[0])
        results.append({
            'cls': cls,
            'metric': 'auc',
            'value': report2.loc[cls, 'auroc'],
            'diff': report1.loc[cls, 'auroc'] - report2.loc[cls, 'auroc'],
            'p': p,
        })

    for metric in metrics_list[1:]:
        results.append({
            'cls': 'macro',
            'metric': metric,
            'value': report2.loc['macro', report_metric_map(metric)],
            # no p value for macro-avg metrics
        })

    df = pd.DataFrame.from_records(results)
    df['ci-0'] = df.apply(lambda r: ci[get_cls_index(r['cls']), metrics_list.index(r['metric']), 0], axis=1)
    df['ci-1'] = df.apply(lambda r: ci[get_cls_index(r['cls']), metrics_list.index(r['metric']), 1], axis=1)
    df['diff'] = df.apply(lambda r: report1.loc[r['cls'], report_metric_map(r['metric'])] - r['value'], axis=1)
    return df

def main():
    for split in ['test', 'val']:
        for i, target in enumerate(['radiomics', 'scratch']):
            results_df = compare_predictions(split, 'mbmf', target)
            with pd.ExcelWriter(f'MB-data/cmp/{split}.xlsx', mode='a' if i else 'w', engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name=target, index=False)

if __name__ == '__main__':
    main()
