from collections.abc import Callable
import itertools

import pandas as pd
import torch
from tqdm.contrib.concurrent import process_map

import monai
from monai.metrics import compute_meandice
from monai.utils import GridSampleMode
from umei.utils import UMeIParser

from mbs.args import MBSegPredArgs
from mbs.datamodule import load_cohort
from mbs.utils.enums import MBDataKey, SegClass

loader: Callable
args: MBSegPredArgs

def process(data: dict):
    data = loader(data)
    prob = torch.load(args.p_output_dir / data[MBDataKey.CASE] / 'seg-prob.pt', map_location='cpu')
    th = 0.5
    ret = {
        'case': data[MBDataKey.CASE],
        'split': data['split'],
    }
    for i, seg_class in enumerate(args.seg_classes):
        pred = (prob[:, i:i + 1] > th).long()
        seg = data[seg_class][None]
        ret[f'dice-{seg_class}'] = compute_meandice(pred, seg).item()
        ret[f'recall-{seg_class}'] = ((pred * seg).sum() / seg.sum()).item()
    return ret

def main():
    global args, loader
    parser = UMeIParser((MBSegPredArgs, ), use_conf=True)
    args = parser.parse_args_into_dataclasses()[0]
    seg_keys = list(SegClass)
    loader = monai.transforms.Compose([
        monai.transforms.LoadImageD(seg_keys),
        monai.transforms.EnsureChannelFirstD(seg_keys),
        monai.transforms.OrientationD(seg_keys, axcodes='RAS'),
        monai.transforms.SpacingD(seg_keys, pixdim=args.spacing, mode=GridSampleMode.NEAREST),
        monai.transforms.ResizeWithPadOrCropD(seg_keys, spatial_size=args.pad_crop_size),
        monai.transforms.LambdaD(seg_keys, lambda x: x.as_tensor()),
    ])
    cohort = load_cohort()
    for split, data in cohort.items():
        for x in data:
            x['split'] = split
    results = process_map(process, list(itertools.chain(*cohort.values())), max_workers=43)
    pd.DataFrame.from_records(results).to_excel(args.p_output_dir / 'eval.xlsx')
    pd.DataFrame.from_records(results).to_excel(args.p_output_dir / 'eval.csv')

if __name__ == '__main__':
    main()
