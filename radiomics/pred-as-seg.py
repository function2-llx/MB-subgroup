from pathlib import Path

import toolz

from luolib.utils import DataKey
from monai import transforms as monai_t
import nibabel as nib
import torch
from torch.nn import functional as torch_f

from mbs.utils.enums import SegClass, Modality
from monai.data import MetaTensor
from monai.utils import TraceKeys

data_dir = Path('../mbs/processed/cr-p10/register-crop')
pred_output_dir = Path('../output/seg/base-3l/predict-42/sw0.5+tta/pred')
output_dir = Path('pred-seg')
output_dir.mkdir(exist_ok=True)

def main():
    transform = monai_t.Compose([
        monai_t.LoadImageD(list(Modality), image_only=True, ensure_channel_first=True),
        monai_t.ToDeviceD(list(Modality), f'cuda'),
        monai_t.ConcatItemsD(list(Modality), DataKey.IMG),
        monai_t.Lambda(lambda data: data[DataKey.IMG]),
        monai_t.Orientation('RAS'),
        monai_t.ScaleIntensityRangePercentiles(0.5, 99.5, b_min=0, b_max=1, clip=True, channel_wise=True),
        monai_t.CropForeground(),
    ])
    for case in ['483517', '586779']:
        img: MetaTensor = transform({
            modality: data_dir / case / f'{modality}.nii'
            for modality in Modality
        })

        crop_op = img.applied_operations[-1]
        pred = torch.load(pred_output_dir / case / 'seg-prob.pt', map_location='cuda')
        pred = pred[:, list(SegClass).index(SegClass.AT)]
        pred = pred[:, None]  # add channel dim
        pred = (torch_f.interpolate(pred, img.shape[1:], mode='trilinear') > 0.5).byte()
        to_pad = list(toolz.partition(2, crop_op[TraceKeys.EXTRA_INFO]['cropped']))
        pred = torch_f.pad(pred, list(toolz.concat(reversed(to_pad))))
        pred = pred[0, 0]
        original_code = nib.orientations.aff2axcodes(img.meta['original_affine'])
        flip_dims = [
            i for i in range(3)
            if original_code[i] != 'RAS'[i]
        ]
        pred = torch.flip(pred, flip_dims)
        pred_nib = nib.Nifti1Image(pred.cpu().numpy(), img.meta['original_affine'])
        nib.save(pred_nib, output_dir / f'{case}.nii')

if __name__ == '__main__':
    main()
