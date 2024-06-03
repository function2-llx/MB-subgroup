from pathlib import Path

import numpy as np
import pandas as pd

from mbs.datamodule import load_merged_plan
from mbs.utils.enums import SUBGROUPS

def main():
    dfs = pd.read_excel('radiomics/prob-results.xlsx', sheet_name=['s1', 's2-ws', 's2-g34'], dtype={'case': 'string'})
    for df in dfs.values():
        df.set_index('case', inplace=True)
    index = dfs['s1'].index
    plan = load_merged_plan()
    y_true = plan.loc[index, 'subgroup']
    output_dir = Path('radiomics/case-output')
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in ['SVM', 'LR', 'RF', 'XGB', 'MLP']:
        g34 = dfs['s1'][model].to_numpy()
        shh = dfs['s2-ws'][model].to_numpy()
        g4 = dfs['s2-g34'][model].to_numpy()
        prob = np.stack([(1 - g34) * (1 - shh), (1 - g34) * shh, g34 * (1 - g4), g34 * g4], axis=-1)
        df = pd.DataFrame(prob, index, SUBGROUPS)
        df['split'] = 'test'
        df['pred'] = list(map(lambda subgroup_idx: SUBGROUPS[subgroup_idx], prob.argmax(axis=-1).tolist()))
        df['true'] = y_true
        df.to_excel(output_dir / f'{model}.xlsx', sheet_name='4way')

if __name__ == '__main__':
    main()
