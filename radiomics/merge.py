from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

from mbs.utils.enums import MBDataKey
from mbs.datamodule import DATA_DIR, parse_age

PARENT = Path(__file__).parent

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', default='extractive.csv')
    parser.add_argument('-o', default='features.csv')
    args = parser.parse_args()
    df = pd.read_csv(PARENT / args.i).drop(columns=['Image', 'Mask'])
    df['住院号'] = df[MBDataKey.CASE].map(lambda x: x[:6])
    df = df.set_index('住院号')
    clinical = pd.read_excel(DATA_DIR / 'clinical-com.xlsx', dtype=str).set_index('住院号')
    shared_names = {MBDataKey.CASE, 'molecular', 'split'}
    ret = pd.concat(
        [
            df[df['modality'] == modality]
            .drop(columns='modality')
            .rename(columns=lambda x: x if x in shared_names else f'{modality}_{x}')
            for modality in ['t1', 't2']
        ],
        axis='columns',
    )

    ret.insert(2, 'sex', clinical['sex'])
    ret.insert(3, 'age', clinical['age'].map(parse_age))

    ret.loc[:, ~ret.columns.duplicated()].to_csv(PARENT / args.o)

if __name__ == '__main__':
    main()
