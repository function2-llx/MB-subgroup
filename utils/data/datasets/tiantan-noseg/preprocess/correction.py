import json
from pathlib import Path

import SimpleITK as sitk
from tqdm.contrib.concurrent import process_map

from skull_stripping import output_dir as input_dir
from utils.dicom_utils import ScanProtocol

output_dir = Path('n4itk')

def correct(info):
    patient = info['patient']
    save_dir = output_dir / patient
    save_dir.mkdir(parents=True, exist_ok=True)
    for protocol in ScanProtocol:
        img = sitk.ReadImage(str(input_dir / patient / f'{protocol.name}.nii.gz'), sitk.sitkFloat32)
        corrected = sitk.N4BiasFieldCorrection(img)
        sitk.WriteImage(corrected, str(save_dir / f'{protocol.name}.nii.gz'))

if __name__ == '__main__':
    cohort = json.load(open('cohort.json'))
    process_map(correct, cohort, ncols=80, max_workers=4)
