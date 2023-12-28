from pathlib import Path

import cytoolz
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as nnf

from luolib.transforms import nnUNetLoader
from luolib.utils import process_map
from monai import transforms as mt
from monai.data import MetaTensor

from mbs.datamodule import load_merged_plan

data_dir = Path('MB-data/processed/cr-p10/register-crop')
seg_output_dir = Path('MB-data/seg-pred/(16, 256, 256)+0.8+gaussian+tta')
plot_dir = Path('seg-plot')
PRED_TH = 0.3
meta_loader = nnUNetLoader(Path('nnUNet_data/preprocessed/Dataset500_TTMB/nnUNetPlans-z_3d_fullres'), None, None)
plan = load_merged_plan()
img_keys = ['T2', 'ST', 'AT']
img_loader = mt.Compose([
    mt.LoadImageD(img_keys, ensure_channel_first=True),
    mt.OrientationD(img_keys, 'SPR'),
    mt.ToNumpyD(['T2', 'ST', 'AT']),
])
dpi = 100

def process(case: str):
    meta = meta_loader(case)
    if not (data_dir / case / f'AT.nii').exists():
        return
    data: dict[str, np.ndarray] = img_loader({
        key: data_dir / case / f'{key}.nii'
        for key in ['T2', 'ST', 'AT']
    })
    prob = torch.load(seg_output_dir / case / 'prob.pt', 'cpu')
    if isinstance(prob, MetaTensor):
        prob = prob.as_tensor()
    prob = nnf.interpolate(
        prob[None],
        meta['shape_after_cropping_and_before_resampling'],
        mode='trilinear',
    )[0]
    pred = prob > PRED_TH
    origin_shape = meta['shape_before_cropping']
    bbox = meta['bbox_used_for_cropping']
    padding = [
        (bbox[i][0], origin_shape[i] - bbox[i][1])
        for i in range(3)
    ]
    pred = nnf.pad(pred, [*cytoolz.concat(reversed(padding))])
    figsize = (pred.shape[3] / dpi, pred.shape[2] / dpi)
    sitk_stuff = meta['sitk_stuff']
    direction = np.array(sitk_stuff['direction']).reshape((3, 3))
    affine = np.eye(4)
    affine[:3, :3] = direction[[2, 1, 0]]
    affine[1] = -affine[1]
    affine[2] = -affine[2]
    pred = MetaTensor(pred, affine)
    pred = mt.Orientation('SPR')(pred).byte().cpu().numpy()
    for i in data['AT'][0].sum(axis=(1, 2)).argsort()[-5:]:
        case_plot_dir = plot_dir / plan.at[case, 'group'] / plan.at[case, 'subgroup'] / case / str(i)
        case_plot_dir.mkdir(exist_ok=True, parents=True)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig: Figure
        ax: Axes
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        ax.imshow(data['T2'][0, i], cmap='gray')
        fig.savefig(case_plot_dir / 'origin.png')
        ax.imshow(data['ST'][0, i], ListedColormap(['none', 'green']), alpha=0.3)
        ax.imshow(data['AT'][0, i], ListedColormap(['none', 'red']), alpha=0.3)
        fig.savefig(case_plot_dir / 'label.png')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        ax.imshow(data['T2'][0, i], cmap='gray')
        for c, color in enumerate(['green', 'red']):
            ax.imshow(pred[c, i], ListedColormap(['none', color]), alpha=0.3)
        fig.savefig(case_plot_dir / 'pred.png')
        plt.close(fig)

def main():
    plot_dir.mkdir(exist_ok=True, parents=True)
    process_map(process, plan.index, max_workers=0)

if __name__ == '__main__':
    main()
