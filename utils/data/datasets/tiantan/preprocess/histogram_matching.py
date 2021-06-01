import json
from pathlib import Path

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from tqdm.contrib.concurrent import process_map

from correction import output_dir as input_dir
from utils.dicom_utils import ScanProtocol

output_dir = Path('matched')
protocol: ScanProtocol
ref: sitk.Image

def get_range(info):
    patient = info['patient']
    data = nib.load(input_dir / patient / f'{protocol.name}.nii.gz').get_fdata()
    return data.max() - data.min()

def match(info):
    patient = info['patient']
    save_dir = output_dir / patient
    save_dir.mkdir(parents=True, exist_ok=True)
    for protocol in ScanProtocol:
        img = sitk.ReadImage(str(input_dir / patient / f'{protocol.name}.nii.gz'), sitk.sitkFloat32)
        # HistogramMatchingImageFilter does not support `SetReferenceImage` in sitk (2.1)
        # do not have time to discover how ITK Python API works
        # so sad :(
        matched = sitk.HistogramMatching(img, ref)
        sitk.WriteImage(matched, str(save_dir / f'{protocol.name}.nii.gz'))

if __name__ == '__main__':
    cohort = json.load(open('cohort.json'))
    for protocol in ScanProtocol:
        print(f'working for {protocol.name}')
        ranges = process_map(get_range, cohort, ncols=80, desc='finding reference')
        print('range mean:', np.mean(ranges))
        # pick the patient has the median intensity range as ref for this protocol
        ref_id = np.argpartition(ranges, len(ranges) // 2)[len(ranges) // 2]
        ref_patient = cohort[ref_id]['patient']
        ref = sitk.ReadImage(str(input_dir / ref_patient / f'{protocol.name}'))
        print(ref_patient, 'range:', ranges[ref_id])
        process_map(match, cohort, ncols=80, desc='matcing')
