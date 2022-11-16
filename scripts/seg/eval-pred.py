from collections.abc import Callable
import itertools

import pandas as pd
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
    # prob = torch.load(args.p_output_dir / data[MBDataKey.CASE] / SEG_PROB_FILENAME, map_location='cpu')
    ret = {
        'case': data[MBDataKey.CASE],
        'split': data['split'],
    }
    st_pred = data[f'{SegClass.ST}-pred'][None]
    for i, seg_class in enumerate(args.seg_classes):
        seg = data[seg_class][None]
        pred = data[f'{seg_class}-pred'][None]
        ret[f'dice-{seg_class}'] = compute_meandice(pred, seg).item()
        ret[f'recall-{seg_class}'] = ((pred * seg).sum() / seg.sum()).item()
        if seg_class != SegClass.ST:
            ret[f'recall-{seg_class}-st'] = ((st_pred * seg).sum() / seg.sum()).item()
    bbox = data[f'{SegClass.ST}-pred_bbox'][0]
    for i in range(3):
        ret[f'b{i}'] = bbox[i << 1 | 1] - bbox[i << 1]
    return ret

def main():
    global args, loader
    parser = UMeIParser((MBSegPredArgs, ), use_conf=True)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    all_keys = list(args.seg_classes) + list(map(lambda x: f'{x}-pred', args.seg_classes))

    loader = monai.transforms.Compose([
        monai.transforms.LoadImageD(all_keys),
        monai.transforms.EnsureChannelFirstD(all_keys),
        monai.transforms.OrientationD(all_keys, axcodes='RAS'),
        monai.transforms.SpacingD(all_keys, pixdim=args.spacing, mode=GridSampleMode.NEAREST),
        monai.transforms.ResizeWithPadOrCropD(all_keys, spatial_size=args.pad_crop_size),
        monai.transforms.BoundingRectD(f'{SegClass.ST}-pred'),
        monai.transforms.LambdaD(all_keys, lambda x: x.as_tensor()),
    ])
    cohort = load_cohort()
    suffix = f'th{args.th}'
    if args.do_post:
        suffix += '-post'
    print(suffix)
    for split, data in cohort.items():
        for x in data:
            x['split'] = split
            for seg_class in args.seg_classes:
                x[f'{seg_class}-pred'] = args.p_output_dir / x[MBDataKey.CASE] / suffix / f'{seg_class}.nii.gz'
    results = process_map(process, list(itertools.chain(*cohort.values())), max_workers=43)
    results_df = pd.DataFrame.from_records(results)
    results_df.to_excel(args.p_output_dir / f'eval-{suffix}.xlsx', index=False)
    results_df.to_csv(args.p_output_dir / f'eval-{suffix}.csv', index=False)

if __name__ == '__main__':
    main()
