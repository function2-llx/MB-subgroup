from argparse import ArgumentParser
import itertools as it

import cytoolz
import nipype
from nipype.interfaces.niftyreg import RegAladin, RegResample
from nipype.interfaces.niftyreg.reg import RegAladinInputSpec
import pandas as pd
import torch
from tqdm.contrib.concurrent import process_map

import monai
from monai import transforms as monai_t
from monai.data import MetaTensor
from monai.utils import GridSampleMode

from mbs.datamodule import DATA_DIR, PROCESSED_DIR, SEG_REF
from mbs.utils.enums import MBDataKey, Modality, SegClass

spacing = (0.468750000, 0.468751997, 6.499997139)
output_dir = PROCESSED_DIR / 'resampled'
output_dir.mkdir(exist_ok=True)
plan = pd.read_excel(PROCESSED_DIR / 'plan.xlsx', sheet_name='merge', dtype={MBDataKey.NUMBER: 'string'})
plan.set_index(MBDataKey.NUMBER, inplace=True)
assert plan.index.unique().size == plan.index.size

writer = monai.data.NibabelWriter()

def adjust_spacing(number: str, image_type: str, cuda_id: int):
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

from nipype.interfaces.base import traits

# 懂的都懂
class MyRegAladinInputSpec(RegAladinInputSpec):
    pad_val = traits.Float(desc="Padding value", argstr="-pad %f")

class MyRegAladin(RegAladin):
    input_spec = MyRegAladinInputSpec

def register(number: str, cuda_id: int):
    node = MyRegAladin()
    case_data_dir = DATA_DIR / plan.at[number, 'group'] / plan.at[number, 'name']
    case_output_dir = output_dir / number
    inputs: node.input_spec = node.inputs
    inputs.ref_file = case_output_dir / f'{Modality.T2}.nii'
    inputs.platform_val = 1
    inputs.gpuid_val = cuda_id
    inputs.pad_val = 0
    inputs.verbosity_off_flag = True
    for modality in [Modality.T1, Modality.T1C]:
        img_path = case_data_dir / f'{modality}.nii'
        inputs.flo_file = img_path
        inputs.aff_file = case_output_dir / f'{modality}-reg.txt'
        inputs.res_file = case_output_dir / f'{modality}.nii'
        node.run()
    node = RegResample()
    if (ct_file_path := case_data_dir / f'{SegClass.CT}.nii').exists():
        inputs: node.input_spec = node.inputs
        inputs.ref_file = case_output_dir / f'{Modality.T2}.nii'
        inputs.flo_file = ct_file_path
        inputs.out_file = case_output_dir / f'{SegClass.CT}.nii'
        inputs.trans_file = case_output_dir / f'{SEG_REF[SegClass.CT]}-reg.txt'
        inputs.inter_val = 'NN'
        inputs.verbosity_off_flag = True
        node.run()

def main():
    parser = ArgumentParser()
    parser.add_argument('--adjust_workers', type=int, default=1)
    parser.add_argument('--register_workers', type=int, default=8)

    args = parser.parse_args()
    # adjust_types = [Modality.T2, *[seg_class for seg_class, modality in SEG_REF.items() if modality == Modality.T2]]
    # process_map(
    #     adjust_spacing,
    #     cytoolz.interleave(it.repeat(plan.index, len(adjust_types))),
    #     it.cycle(adjust_types),
    #     it.cycle(range(torch.cuda.device_count())),
    #     total=plan.index.size * len(adjust_types),
    #     max_workers=args.adjust_workers,
    #     dynamic_ncols=True,
    #     desc='adjust spacing of T2, AT, ST'
    # )
    process_map(
        register,
        plan.index,
        it.cycle(range(torch.cuda.device_count())),
        total=plan.index.size,
        dynamic_ncols=True,
        max_workers=args.register_workers,
        chunksize=1,
        desc='co-registering T1, T1C, CT',
    )

if __name__ == '__main__':
    main()
