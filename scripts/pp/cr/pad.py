from argparse import ArgumentParser
from pathlib import Path

import ants
import pandas as pd
from tqdm.contrib.concurrent import process_map

from mbs.datamodule import DATA_DIR, PROCESSED_DIR
from mbs.utils.enums import MBDataKey, Modality, SegClass

plan = pd.read_excel(PROCESSED_DIR / 'plan.xlsx', sheet_name='merge', dtype={MBDataKey.NUMBER: 'string'})
plan.set_index(MBDataKey.NUMBER, inplace=True)
assert plan.index.unique().size == plan.index.size

output_dir: Path
padding: int

def pad(number: str):
    case_data_dir = DATA_DIR / plan.at[number, 'group'] / plan.at[number, 'name']
    case_output_dir = output_dir / number
    case_output_dir.mkdir(exist_ok=True)
    orientation = ants.get_orientation(ants.image_read(str(case_data_dir / f'{Modality.T2}.nii')))

    for img_type in [*Modality, *SegClass]:
        if not (src := case_data_dir / f'{img_type}.nii').exists():
            continue
        img = ants.image_read(str(src)).reorient_image2(orientation)
        img = ants.pad_image(img, pad_width=((padding, padding), ) * img.dimension)
        ants.image_write(img, str(case_output_dir / f'{img_type}.nii'))

def main():
    global output_dir, padding
    parser = ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--padding', type=int, default=10)
    args = parser.parse_args()
    padding = args.padding
    output_dir = PROCESSED_DIR / f'cr-p{padding}' / f'pad'
    output_dir.mkdir(exist_ok=True, parents=True)

    process_map(
        pad,
        plan.index,
        total=plan.index.size,
        dynamic_ncols=True,
        max_workers=args.workers,
        desc='padding',
    )

if __name__ == '__main__':
    main()
