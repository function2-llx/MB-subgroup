from enum import StrEnum

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from statsmodels.stats.contingency_tables import mcnemar

from mbs.utils.enums import SUBGROUPS
from _misc import read_case_outputs

pROC = importr('pROC')

class Metric(StrEnum):
    SEN = 'sensitivity'
    SPE = 'specificity'
    ACC = 'accuracy'

def mcnemar_table(p1: np.ndarray, p2: np.ndarray):
    table = np.array([
        [(p1 & p2).sum(), (~p1 & p2).sum()],
        [(p1 & ~p2).sum(), (~p1 & ~p2).sum()],
    ])

    return table, table[0, 1] + table[1, 0] < 25

def main():
    rng = np.random.default_rng(42)
    o1 = read_case_outputs('case-output.xlsx')
    # o2 = read_case_outputs('case-output-scratch.xlsx')
    o2 = read_case_outputs('radiomics/case-output/MLP.xlsx')
    assert ((true := o1['true'].to_numpy()) == o2['true'].to_numpy()).all()
    p1 = o1['pred'].to_numpy()
    p2 = o2['pred'].to_numpy()
    results = []
    for cls in SUBGROUPS:
        cls_mask = true == cls
        p1b = (p1 == cls) == cls_mask
        p2b = (p2 == cls) == cls_mask
        for metric in Metric:
            match metric:
                case Metric.SEN:
                    sub_idx = cls_mask
                case Metric.SPE:
                    sub_idx = ~cls_mask
                case Metric.ACC:
                    sub_idx = slice(None)
            # sub_true = true[sub_idx]
            table, exact = mcnemar_table(p1b[sub_idx], p2b[sub_idx])
            result = mcnemar(table, exact=False, correction=table[1, 0] == table[0, 1])
            p = result.pvalue
            # print(table)
            print(cls, metric, result.statistic, p)
            results.append({
                'cls': cls,
                'metric': metric,
                'p': p,
            })
        true_r = robjects.FloatVector(cls_mask)
        prob1_r = robjects.FloatVector(o1[cls].to_numpy())
        prob2_r = robjects.FloatVector(o2[cls].to_numpy())
        roc_1 = pROC.roc(true_r, prob1_r)
        roc_2 = pROC.roc(true_r, prob2_r)
        test_result = pROC.roc_test(roc_1, roc_2, method='delong')
        print('auc', p := test_result.rx2('p.value')[0])
        results.append({
            'cls': cls,
            'metric': 'auc',
            'p': p,
        })
    pd.DataFrame.from_records(results).to_excel('stat-results.xlsx')

if __name__ == '__main__':
    main()
