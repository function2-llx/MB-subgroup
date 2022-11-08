from pathlib import Path

import pandas as pd

from mbs.utils.enums import MBDataKey

PARENT = Path(__file__).parent

def main():
    df = pd.read_csv(PARENT / 'extractive.csv').drop(columns=['Image', 'Mask'])
    shared_names = {MBDataKey.CASE, 'molecular', 'split'}
    pd.concat(
        [
            df[df['modality'] == modality].set_index(MBDataKey.CASE, drop=True)
            .drop(columns='modality')
            .rename(columns=lambda x: x if x in shared_names else f'{modality}_{x}')
            for modality in ['t1', 't2']
        ],
        axis='columns',
    ).T.drop_duplicates().T.to_csv(PARENT / 'features.csv')

if __name__ == '__main__':
    main()
