import itertools

import pandas as pd
from tqdm.contrib.concurrent import process_map

import monai
from monai.utils import GridSampleMode

from mbs.utils.enums import MBDataKey, Modality, SegClass
from mbs.datamodule import DATA_DIR, load_split_cohort

img_keys = list(Modality)
# seg_keys = list(SegClass)
all_keys = img_keys

cohort = list(itertools.chain(*load_split_cohort().values()))

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(all_keys),
    monai.transforms.EnsureChannelFirstD(all_keys),
    monai.transforms.OrientationD(all_keys, axcodes='RAS'),
    monai.transforms.SpacingD(img_keys, pixdim=[0.46875, 0.46875, -1], mode=GridSampleMode.BILINEAR),
    # monai.transforms.SpacingD(seg_keys, pixdim=[0.46875, 0.46875, -1], mode=GridSampleMode.NEAREST),
])

def check(data):
    data = loader(data)
    return {
        'case': data[MBDataKey.CASE],
        **{
            k: data[k].exists()
            for k in SegClass
        },
        **{
            k: data[k].shape
            for k in Modality
        },
    }

def main():
    results = process_map(check, cohort, ncols=80, max_workers=16)
    pd.DataFrame.from_records(results).to_excel(DATA_DIR / 'check-stack.xlsx', index=False)

if __name__ == '__main__':
    main()
