import pandas as pd

from mbs.datamodule import DATA_DIR, parse_age

clinical = pd.read_excel(DATA_DIR / 'clinical-com.xlsx', index_col='住院号')

def check_adult():
    label = pd.read_excel(DATA_DIR / 'subgroup-20230105-核对.xlsx', sheet_name='成人', index_col='No.')
    for number, (name, subgroup, number_pinyin, sex, age) in label.iterrows():
        if number not in clinical.index:
            continue
        if sex != clinical.at[number, 'sex']:
            print(number, name, sex, clinical.at[number, 'sex'])
        if age != clinical.at[number, 'age']:
            print(number, name, age, clinical.at[number, 'age'])
        # print(number, name, subgroup, number_pinyin, sex, age)

def check_child():
    label = pd.read_excel(DATA_DIR / 'subgroup-20230105-核对.xlsx', sheet_name='儿童')
    for _, (name, number, subgroup, sex, age) in label.iterrows():
        if number not in clinical.index:
            continue
        if not pd.isna(sex) and sex != clinical.at[number, 'sex']:
            print(number, name, sex, clinical.at[number, 'sex'])
        parsed_age = parse_age(clinical.at[number, 'age'])
        if not pd.isna(age) and age != parsed_age:
            print(number, name, age, clinical.at[number, 'age'])

def main():
    # check_adult()
    check_child()

if __name__ == '__main__':
    main()
