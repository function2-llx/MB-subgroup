import pandas as pd

from mbs.datamodule import load_merged_plan
from mbs.utils.enums import DATA_DIR

meta = pd.read_excel(DATA_DIR / '影像预测分子分型.xlsx', dtype={'number': 'string'}).set_index('number')
meta.rename(columns={'gender': 'sex'}, inplace=True)
plan = load_merged_plan()

def main():
    for number, case in meta.iterrows():
        age_group = 'adult' if (age := case['age']) >= 18 else 'child'
        if plan.at[number, 'group'] != age_group and age != 18:
            print(number)
        if plan.at[number, 'subgroup'] != case['subgroup']:
            print(number)

if __name__ == '__main__':
    main()
