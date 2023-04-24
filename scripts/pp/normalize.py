from argparse import ArgumentParser
import itertools as it
from pathlib import Path

import numpy as np
import torch
from tqdm.contrib.concurrent import process_map

from luolib.utils import DataKey
import monai
from monai import transforms as monai_t
from monai.utils import GridSampleMode

from mbs.datamodule import load_merged_plan
from mbs.utils.enums import Modality, SegClass

spacing = (0.46875, 0.468751997, 6.499997139)
plan = load_merged_plan()
output_dir: Path
data_dir: Path
writer = monai.data.NibabelWriter()

def normalize(number: str, cuda_id: int):
    case_data_dir = data_dir / number
    case_output_dir = output_dir / number
    case_output_dir.mkdir(exist_ok=True)
    all_keys = [*Modality, *SegClass]
    no_seg = all(not (case_data_dir / f'{seg_class}.nii').exists() for seg_class in SegClass)
    transform = monai_t.Compose([
        monai_t.LoadImageD(all_keys, image_only=True, ensure_channel_first=True, allow_missing_keys=no_seg),
        monai_t.ToDeviceD(all_keys, f'cuda:{cuda_id}', allow_missing_keys=no_seg),
        monai_t.OrientationD(all_keys, 'RAS', allow_missing_keys=True),
        monai_t.ConcatItemsD(list(Modality), DataKey.IMG),
        *(() if no_seg else (monai_t.ConcatItemsD(list(SegClass), DataKey.SEG),)),
        monai_t.ScaleIntensityRangePercentilesD(DataKey.IMG, 0.5, 99.5, b_min=0, b_max=1, clip=True, channel_wise=True),
        monai_t.CropForegroundD([DataKey.IMG, DataKey.SEG], source_key=DataKey.IMG, allow_missing_keys=no_seg),
        monai_t.SpacingD(
            [DataKey.IMG, DataKey.SEG],
            pixdim=(*spacing[:2], -1),
            mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
            allow_missing_keys=no_seg,
        ),
        monai_t.NormalizeIntensityD(DataKey.IMG, nonzero=True, set_zero_to_min=True),
        monai_t.SelectItemsD([DataKey.IMG, DataKey.SEG], allow_missing_keys=no_seg),
    ])
    data: dict = transform({
        img_type: img_path
        for img_type in all_keys if (img_path := case_data_dir / f'{img_type}.nii').exists()
    })
    for key, img in data.items():
        np.save(case_output_dir / f'{key}.npy', data[key])

def main():
    global data_dir, output_dir
    parser = ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--data_dir', type=Path, default=Path('mbs/processed/cr-p10/register-crop'))
    args = parser.parse_args()
    data_dir = args.data_dir

    output_dir = data_dir.parent / 'normalized'
    output_dir.mkdir(exist_ok=True)

    process_map(
        normalize,
        plan.index,
        it.cycle(range(torch.cuda.device_count())),
        total=plan.index.size,
        dynamic_ncols=True,
        max_workers=args.workers,
    )

if __name__ == '__main__':
    main()
