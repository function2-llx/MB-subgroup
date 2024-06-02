import pandas as pd

from luolib.utils import process_map
from monai import transforms as mt

from mbs.datamodule import load_merged_plan, load_split
from mbs.utils.enums import PROCESSED_DIR
from monai.data import MetaTensor
split = load_split()
plan = load_merged_plan()

data_dir = PROCESSED_DIR / 'cr-p10' / 'register-crop'
loader = mt.Compose([
    mt.LoadImage('NibabelReader', ensure_channel_first=True),
    mt.Orientation('SRA'),
])

def solve(case: str):
    if not (path := data_dir / case / 'AT.nii').exists():
        return {
            'case': case,
            'split': split[case],
            'volume': None,
        }
    img: MetaTensor = loader(path)
    volume = img.sum() * img.pixdim.prod()
    return {
        'case': case,
        'split': split[case],
        'volume': volume.item(),
    }

def main():
    print(plan)
    pd.DataFrame.from_records(
        process_map(solve, plan.index, max_workers=16),
        'case',
    ).to_excel('tumor-stat.xlsx')

if __name__ == '__main__':
    main()
