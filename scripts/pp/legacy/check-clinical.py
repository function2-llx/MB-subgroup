import pandas as pd

from mbs.datamodule import DATA_DIR

def main():
    cohort = pd.read_excel(DATA_DIR / 'plan-split.xlsx', sheet_name='Sheet1').set_index('name')
    clinical = pd.read_excel(DATA_DIR / 'clinical.xlsx').set_index('name')
    for name in cohort.index:
        if name not in clinical.index:
            print(name)

if __name__ == '__main__':
    main()
