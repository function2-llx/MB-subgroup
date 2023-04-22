from argparse import ArgumentParser
import itertools as it
import os
from pathlib import Path

import ants
from nipype.interfaces.base import traits
from nipype.interfaces.niftyreg import RegAladin, RegResample
from nipype.interfaces.niftyreg.reg import RegAladinInputSpec
import numpy as np
from tqdm.contrib.concurrent import process_map

from mbs.datamodule import load_merged_plan
from mbs.utils.enums import MBDataKey, Modality, SegClass, PROCESSED_DIR, DATA_DIR, SEG_REF

plan = load_merged_plan()
data_dir: Path
output_dir: Path
cropped_output_dir: Path
padding: int

# 懂的都懂
class MyRegAladinInputSpec(RegAladinInputSpec):
    pad_val = traits.Float(desc="Padding value", argstr="-pad %f")

class MyRegAladin(RegAladin):
    input_spec = MyRegAladinInputSpec

def crop(number: str, img_type: str):
    case_output_dir = cropped_output_dir / number
    case_output_dir.mkdir(exist_ok=True)

    img: ants.ANTsImage = ants.image_read(str(output_dir / number / f'{img_type}.nii'))
    cropped = ants.crop_indices(img, (padding, ) * img.dimension, np.array(img.shape) - padding)
    ants.image_write(cropped, str(case_output_dir / f'{img_type}.nii'))

def register(number: str, cuda_id: int):
    # niftyreg will ocupy the first and the specified card, I have to do this（摊手）
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
    case_origin_dir = DATA_DIR / plan.at[number, MBDataKey.GROUP] / plan.at[number, 'name']
    case_data_dir = data_dir / number
    case_output_dir = output_dir / number
    case_output_dir.mkdir(exist_ok=True)
    ref_path = case_data_dir / f'{Modality.T2}.nii'
    node = MyRegAladin()
    inputs: node.input_spec = node.inputs
    inputs.ref_file = ref_path
    inputs.platform_val = 1
    inputs.gpuid_val = 0
    inputs.pad_val = 0
    inputs.verbosity_off_flag = True
    for modality in [Modality.T1, Modality.T1C]:
        inputs.flo_file = case_data_dir / f'{modality}.nii'
        inputs.aff_file = case_output_dir / f'{modality}-reg.txt'
        inputs.res_file = case_output_dir / f'{modality}.nii'
        node.run()
        crop(number, modality)

    node = RegResample()
    if (ct_file_path := case_data_dir / f'{SegClass.CT}.nii').exists():
        inputs: node.input_spec = node.inputs
        inputs.ref_file = ref_path
        inputs.flo_file = ct_file_path
        inputs.out_file = case_output_dir / f'{SegClass.CT}.nii'
        inputs.trans_file = case_output_dir / f'{SEG_REF[SegClass.CT]}-reg.txt'
        inputs.inter_val = 'NN'
        inputs.verbosity_off_flag = True
        node.run()
        crop(number, SegClass.CT)

    for img_type in [Modality.T2, *[seg_class for seg_class, modality in SEG_REF.items() if modality == Modality.T2]]:
        dst = cropped_output_dir / number / f'{img_type}.nii'
        dst.unlink(missing_ok=True)
        dst.symlink_to(case_origin_dir / f'{img_type}.nii')

def main():
    global data_dir, output_dir, cropped_output_dir, padding
    parser = ArgumentParser()
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--gpu_ids', type=int, nargs='*', default=[0, 1, 2, 3])
    parser.add_argument('--padding', type=int, default=10)
    args = parser.parse_args()

    padding = args.padding
    data_dir = PROCESSED_DIR / f'cr-p{args.padding}' / 'pad'
    output_dir = PROCESSED_DIR / f'cr-p{args.padding}' / 'register-pad'
    cropped_output_dir = PROCESSED_DIR / f'cr-p{args.padding}' / 'final'
    output_dir.mkdir(exist_ok=True)
    cropped_output_dir.mkdir(exist_ok=True)

    process_map(
        register,
        plan.index,
        it.cycle(args.gpu_ids),
        total=plan.index.size,
        dynamic_ncols=True,
        max_workers=args.workers,
        desc='co-registering T1, T1C, CT',
    )

if __name__ == '__main__':
    main()
