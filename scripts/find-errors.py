from pathlib import Path
from collections import Counter
from argparse import ArgumentParser

import pandas as pd

from mbs.conf import cls_map, cls_names
from mbs.datamodule import DATA_DIR

def main():
    parser = ArgumentParser()
    parser.add_argument('check_dir', type=Path, default='.')
    parser.add_argument('--output_path', type=Path, default=None)
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = args.check_dir / 'errors.xlsx'
    cls_scheme = args.check_dir.name
    cmap = cls_map(cls_scheme)
    cnames = cls_names(cls_scheme)
    cohort = pd.read_excel(DATA_DIR / 'plan-split.xlsx', index_col='name')
    cnt = {
        name: Counter()
        for name in cohort.index
    }
    check_dir: Path = args.check_dir
    for results_path in Path(check_dir).rglob('*-results.csv'):
        results = pd.read_csv(results_path, index_col='case')
        for name, info in results.iterrows():
            cnt[name][info['pred']] += 1

    df = pd.DataFrame.from_records([
        {
            'name': name,
            'split': info['split'],
            'subgroup': (subgroup := info['subgroup']),
            **cnt[name],
            'acc':  cnt[name][cnames[cmap[subgroup]]] / s if (s := sum(cnt[name].values())) else None
        }
        for name, info in cohort.iterrows()
    ])
    df = df[~pd.isna(df['acc'])]
    df.to_excel(args.output_path, index=False)
    print(args.output_path.resolve())

if __name__ == "__main__":
    main()
