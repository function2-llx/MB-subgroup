import gzip
from pathlib import Path
import shutil

import nibabel as nib
from tqdm.contrib.concurrent import process_map

from mbs.datamodule import load_merged_plan, load_split
from mbs.utils.enums import Modality, PROCESSED_DIR
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

dataset_name = 'Dataset500_TTMB'
output_dir = Path(nnUNet_raw) / dataset_name

plan = load_merged_plan()
split = load_split()

def convert(case_dir: Path):
    case = case_dir.name
    case_split = 'Ts' if split[case] == 'test' else 'Tr'

    for modality_id, modality in enumerate(Modality):
        with open(case_dir / f'{modality}.nii', 'rb') as src, \
             gzip.open(output_dir / f'images{case_split}' / f'{case}_{modality_id:04d}.nii.gz', 'wb') as dst:
            shutil.copyfileobj(src, dst)
    if case_split == 'Tr':
        st: nib.Nifti1Image = nib.load(case_dir / 'ST.nii')
        at: nib.Nifti1Image = nib.load(case_dir / 'AT.nii')
        seg = st.get_fdata()
        seg[at.get_fdata() == 1] = 2
        nib.save(nib.Nifti1Image(seg, st.affine), output_dir / 'labelsTr' / f'{case}.nii.gz')

def main():
    for dirname in ['imagesTr', 'labelsTr', 'imagesTs']:
        (output_dir / dirname).mkdir(exist_ok=True, parents=True)

    process_map(
        convert, list((PROCESSED_DIR / 'cr-p10/register-crop').iterdir()),
        max_workers=16,
    )

    generate_dataset_json(
        str(output_dir),
        {
            str(i): modality
            for i, modality in enumerate(Modality)
        },
        {
            'background': 0,
            'ST': (1, 2),
            'AT': 2,
        },
        len(list((output_dir / 'labelsTr').iterdir())),
        '.nii.gz',
        (1, 2),
    )

if __name__ == '__main__':
    main()
