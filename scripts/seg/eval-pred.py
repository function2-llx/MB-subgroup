from pathlib import Path

import pandas as pd

from luolib.nnunet import nnUNet_preprocessed
from luolib.reader import PyTorchReader
from luolib.utils import DataKey, device_map, process_map
from luolib import transforms as lt
from mbs.datamodule.seg import ConvertData
from monai import transforms as mt
from monai.metrics import compute_dice

from mbs.datamodule import load_merged_plan, load_split
from mbs.utils.enums import SegClass

data_dir = nnUNet_preprocessed / 'Dataset500_TTMB' / 'nnUNetPlans_3d_fullres'
pred_dir = Path('MB-data/seg-pred/(16, 192, 256)+0.75+gaussian+tta')
pred_th = 0.5

def process(case: str):
    device = device_map()
    seg_path = data_dir / f'{case}_seg.npy'
    has_seg = seg_path.exists()
    # FIXME: separate loaders for seg & pred maybe better, as seg may not exist
    loader = mt.Compose([
        lt.nnUNetLoaderD('case', data_dir, img_key=None, allow_missing=not has_seg),
        *((ConvertData(), ) if has_seg else ()),
        mt.ToDeviceD('seg', device, allow_missing_keys=not has_seg),
        mt.LoadImageD('pred', ensure_channel_first=False, image_only=True, reader=PyTorchReader),
        mt.ToDeviceD('pred', device),
        mt.LambdaD('pred', lambda x: x.as_tensor(), track_meta=False),
        # mt.BoundingRectD(f'{SegClass.ST}-pred'),
    ])
    data = loader({
        'case': case,
        'pred': pred_dir / case / 'prob.pt',
    })
    ret = {'case': case}
    if 'seg' in data:
        seg = data['seg']
        pred = data['pred'] > pred_th
        dice = compute_dice(seg[None], pred[None])[0]
        st_pred = pred[0]
        for i, seg_class in enumerate(['ST', 'AT']):
            ret[f'dice-{seg_class}'] = dice[i].item()
            ret[f'recall-{seg_class}'] = ((pred[i] * seg[i]).sum() / seg[i].sum()).item()
            if seg_class != SegClass.ST:
                ret[f'recall-{seg_class}-st'] = ((st_pred * seg[i]).sum() / seg[i].sum()).item()

    # bbox = data[f'{SegClass.ST}-pred_bbox'][0]
    # center = np.empty(3, dtype=np.int32)
    # for i in range(3):
    #     low = bbox[i << 1]
    #     high = bbox[i << 1 | 1]
    #     ret[f'b{i}'] = high - low
    #     center[i] = low + high >> 1
    # ret['center'] = center
    return ret

def main():
    plan = load_merged_plan()
    # for post in [False, True]:
    results = process_map(
        process,
        plan.index,
        max_workers=4, dynamic_ncols=True,
    )
    results_df = pd.DataFrame(results).set_index('case')
    results_df['split'] = load_split()
    # center_save_path = conf.p_output_dir / 'center' / f'{MBSegPredConf.get_sub(conf, post)}.json'
    # center_save_path.parent.mkdir(exist_ok=True)
    # results_df.pop('center').to_json(center_save_path, force_ascii=False, indent=4)

    with pd.ExcelWriter(
        result_file := pred_dir / f'eval.xlsx',
        mode=(mode := 'a' if result_file.exists() else 'w'),
        if_sheet_exists='replace' if mode == 'a' else None,
    ) as writer:
        results_df.to_excel(writer, f'th={pred_th}', freeze_panes=(1, 0))

if __name__ == '__main__':
    main()
