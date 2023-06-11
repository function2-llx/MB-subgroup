import itertools as it

import numpy as np
import pandas as pd
import torch.cuda
from tqdm.contrib.concurrent import process_map

from luolib.conf import parse_cli
from luolib.reader import PyTorchReader
from luolib.utils import DataKey
import monai
from monai import transforms as mt
from monai.metrics import compute_dice

from mbs.conf import MBSegPredConf
from mbs.datamodule import load_merged_plan, load_split
from mbs.utils.enums import SegClass

conf: MBSegPredConf

def process(case: str, cuda_id: int, post: bool):
    pred_keys = list(map(lambda x: f'{x}-pred', conf.exp_conf.seg_classes))
    all_keys = [DataKey.SEG] + pred_keys
    seg_path = conf.exp_conf.data_dir / case / f'{DataKey.SEG}.npy'
    seg_class_ids = [list(SegClass).index(seg_class) for seg_class in conf.exp_conf.seg_classes]

    no_seg = not seg_path.exists()
    loader = monai.transforms.Compose([
        mt.LoadImageD(DataKey.SEG, ensure_channel_first=False, image_only=True, allow_missing_keys=no_seg),
        mt.LambdaD(DataKey.SEG, lambda x: x[seg_class_ids], allow_missing_keys=no_seg),
        mt.LoadImageD(pred_keys, ensure_channel_first=True, image_only=True, reader=PyTorchReader),
        mt.ToDeviceD(all_keys, f'cuda:{cuda_id}', allow_missing_keys=no_seg),
        mt.ConcatItemsD(pred_keys, 'pred'),
        mt.BoundingRectD(f'{SegClass.ST}-pred'),
        mt.LambdaD(all_keys, lambda x: x.as_tensor(), track_meta=False, allow_missing_keys=no_seg),
    ])

    data = loader({
        **(
            {DataKey.SEG: seg_path} if seg_path.exists()
            else {}
        ),
        **{
            f'{seg_class}-pred': MBSegPredConf.get_save_path(conf, case, seg_class, post)
            for seg_class in conf.exp_conf.seg_classes
        },
    })
    ret = {DataKey.CASE: case}
    if DataKey.SEG in data:
        dice = compute_dice(data[DataKey.SEG][None], data['pred'][None])[0]
        st_pred = data['pred'][list(SegClass).index(SegClass.ST)]
        for i, seg_class in enumerate(conf.exp_conf.seg_classes):
            seg = data[DataKey.SEG][i]
            pred = data['pred'][i]
            ret[f'dice-{seg_class}'] = dice[i].item()
            ret[f'recall-{seg_class}'] = ((pred * seg).sum() / seg.sum()).item()
            if seg_class != SegClass.ST:
                ret[f'recall-{seg_class}-st'] = ((st_pred * seg).sum() / seg.sum()).item()

    bbox = data[f'{SegClass.ST}-pred_bbox'][0]
    center = np.empty(3, dtype=np.int32)
    for i in range(3):
        low = bbox[i << 1]
        high = bbox[i << 1 | 1]
        ret[f'b{i}'] = high - low
        center[i] = low + high >> 1
    ret['center'] = center
    return ret

def main():
    global conf
    conf, _ = parse_cli(MBSegPredConf)
    MBSegPredConf.load_exp_conf(conf)
    MBSegPredConf.default_pred_output_dir(conf)
    plan = load_merged_plan()
    for post in [False, True]:
        results = process_map(
            process,
            plan.index, it.cycle(range(torch.cuda.device_count())), it.repeat(post),
            max_workers=4, dynamic_ncols=True,
        )
        results_df = pd.DataFrame(results).set_index(DataKey.CASE)
        results_df['split'] = load_split()
        center_save_path = conf.p_output_dir / 'center' / f'{MBSegPredConf.get_sub(conf, post)}.json'
        center_save_path.parent.mkdir(exist_ok=True)
        results_df.pop('center').to_json(center_save_path, force_ascii=False, indent=4)

        with pd.ExcelWriter(
            result_file := conf.p_output_dir / f'eval.xlsx',
            mode=(mode := 'a' if result_file.exists() else 'w'),
            if_sheet_exists='replace' if mode == 'a' else None,
        ) as writer:
            results_df.to_excel(writer, sheet_name=MBSegPredConf.get_sub(conf, post), freeze_panes=(1, 0))

if __name__ == '__main__':
    main()
