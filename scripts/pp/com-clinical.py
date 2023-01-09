import pandas as pd

from mbs.datamodule import DATA_DIR

def main():
    clinical = pd.read_excel(DATA_DIR / 'clinical.xlsx')
    clinical['住院号'] = clinical['name'].map(lambda x: x[:6])
    clinical = clinical[~clinical['住院号'].duplicated()].set_index('住院号')
    complement = pd.read_excel('补充.xlsx', dtype={'住院号': str}).set_index('住院号')
    complement.rename(index={'574185': '574158'}, inplace=True)
    for num, (sex, age) in complement.iterrows():
        clinical.loc[num, 'sex'] = {
            '男': 'M',
            '女': 'F',
        }.get(sex, sex)
        if not pd.isna(age):
            clinical.loc[num, 'age'] = f'{round(age):03d}Y'
    clinical.to_excel(DATA_DIR / 'clinical-com.xlsx')

if __name__ == '__main__':
    main()
