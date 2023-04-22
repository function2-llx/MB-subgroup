import itertools as it
from argparse import ArgumentParser

import cytoolz
import pandas as pd
import torch
from tqdm.contrib.concurrent import process_map

from mbs.utils.enums import MBDataKey, Modality, SegClass
import monai
from monai import transforms as monai_t

from mbs.datamodule import DATA_DIR, PROCESSED_DIR
from monai.data import MetaTensor
from monai.utils import GridSampleMode

spacing = (0.46875, 0.468751997, 6.499997139)
output_dir = PROCESSED_DIR / 'resampled'
output_dir.mkdir(exist_ok=True)
plan = pd.read_excel(PROCESSED_DIR / 'plan.xlsx', sheet_name='merge', dtype={MBDataKey.NUMBER: 'string'})
plan.set_index(MBDataKey.NUMBER, inplace=True)
assert plan.index.unique().size == plan.index.size

writer = monai.data.NibabelWriter()

def process(number: str, image_type: str, cuda_id: int):
    name = plan.at[number, 'name']
    image_path = DATA_DIR / plan.at[number, 'group'] / name / f'{image_type}.nii'
    if not image_path.exists():
        return
    sampling_mode = GridSampleMode.BILINEAR if image_type in list(Modality) else GridSampleMode.NEAREST
    transform = monai_t.Compose([
        monai_t.LoadImage(image_only=True, ensure_channel_first=True),
        monai_t.ToDevice(f'cuda:{cuda_id}'),
        monai_t.Orientation('RAS'),
        monai_t.Spacing((*spacing[:2], -1), mode=sampling_mode)
    ])
    data: MetaTensor = transform(image_path)
    writer.set_data_array(data)
    writer.set_metadata(data.meta, resample=False)
    save_path = output_dir / number / f'{image_type}.nii'
    save_path.parent.mkdir(exist_ok=True)
    writer.write(save_path)

def main():
    parser = ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    image_types = [*Modality, *SegClass]
    process_map(
        process,
        cytoolz.interleave(it.repeat(plan.index, len(image_types))),
        it.cycle(image_types),
        it.cycle(range(torch.cuda.device_count())),
        total=plan.index.size * 6,
        max_workers=args.workers,
    )

if __name__ == '__main__':
    main()
