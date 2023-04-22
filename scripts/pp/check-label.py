import pandas as pd

from mbs.datamodule import parse_age
from mbs.utils.enums import MBDataKey, MBGroup, DATA_DIR, PROCESSED_DIR

clinical = pd.read_excel(DATA_DIR / 'clinical' / 'clinical-com.xlsx', dtype={'住院号': 'string'}).set_index('住院号')
label_fix = pd.read_excel(DATA_DIR / 'MB-AX(2).xlsx', dtype={'病历号': 'string'}).set_index('病历号').drop_duplicates()
adult_label = pd.read_excel(DATA_DIR / 'subgroup-20230105-核对.xlsx', sheet_name='成人', dtype={'No.': 'string'}).set_index('No.')
child_label = pd.read_excel(DATA_DIR / 'subgroup-20230105-核对.xlsx', sheet_name='儿童', dtype={'住院号': 'string'})

adult_label.drop(index='269588', inplace=True)  # 手动排除一例分组错误

def check_adult():
    table = pd.DataFrame()
    label = adult_label

    # 手动排除一些例
    label.drop(index=['353168', '380493', '397043'], inplace=True)
    for number in label.index:
        subgroup = label.at[number, 'Subgroup']
        if number not in table.index:
            table.at[number, MBDataKey.SUBGROUP] = subgroup
        elif (subgroup_table := table.at[number, MBDataKey.SUBGROUP]) != subgroup:
            print(number, subgroup_table, subgroup)
        if number in clinical.index:
            age = parse_age(clinical.at[number, 'age'])
            if age < 18:
                print('adult?', number, age)
    table.at['483517', MBDataKey.SUBGROUP] = 'SHH'  # 额外测例
    for number in label_fix.index.intersection(table.index):
        if number in label_fix.index:
            subgroup = table.at[number, MBDataKey.SUBGROUP]
            subgroup_fixed = label_fix.at[number, '分子分型']
            if subgroup != subgroup_fixed:
                print(number, f'{subgroup} -> {subgroup_fixed}')
    return table

def check_child():
    label = child_label
    table = pd.DataFrame()
    for _, (name, number, subgroup, *_) in label.iterrows():
        # 手动排除同时出现在成人组的病例
        if number in ['353168']:
            continue
        if subgroup == 'Group 3':
            subgroup = 'G3'
        if number not in table.index:
            table.at[number, MBDataKey.SUBGROUP] = subgroup
        elif (subgroup_table := table.at[number, MBDataKey.SUBGROUP]) != subgroup:
            print(number, subgroup_table, subgroup)
        if number in clinical.index:
            age = parse_age(clinical.at[number, 'age'])
            if age >= 18:
                print('child?', number, age)
    table.at['586779', MBDataKey.SUBGROUP] = 'WNT'  # 额外测例
    for number in label_fix.index.intersection(table.index):
        if number in label_fix.index:
            subgroup = table.at[number, MBDataKey.SUBGROUP]
            subgroup_fixed = label_fix.at[number, '分子分型']
            if subgroup != subgroup_fixed:
                print(number, f'{subgroup} -> {subgroup_fixed}')
    return table

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(PROCESSED_DIR / 'subgroup.xlsx') as writer:
        adult_table = check_adult()
        child_table = check_child()

        adult_table.to_excel(writer, MBGroup.ADULT, index_label=MBDataKey.CASE)
        child_table.to_excel(writer, MBGroup.CHILD, index_label=MBDataKey.CASE)

    intersection = adult_table.index.intersection(child_table.index)
    if intersection.size > 0:
        print(intersection)

if __name__ == '__main__':
    main()
