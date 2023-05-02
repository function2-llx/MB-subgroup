from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from luolib.utils import DataKey

from mbs.datamodule import parse_age, load_clinical

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', default='extractive.csv', type=Path)
    parser.add_argument('-o', default='features.csv', type=Path)

    args = parser.parse_args()
    df = pd.read_csv(args.i, dtype='string').set_index(DataKey.CASE).drop(columns=['Image', 'Mask'])
    clinical = load_clinical()
    shared_names = {DataKey.CASE, 'molecular', 'split'}
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

    ret.loc[:, ~ret.columns.duplicated()].to_csv(args.o)

if __name__ == '__main__':
    main()
