from pathlib import Path

import monai
import pandas as pd

from mbs.datasets.tiantan import ROOT as data_dir
from mbs.datasets.tiantan.load import read_cohort_info

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(['origin', 'output']),
    monai.transforms.AddChannelD(['origin', 'output']),
])

data_dir = data_dir / 'preprocessed'
output_dir = Path('output/SHHG34/seg/bs8-res34-320-focal/seg-outputs')

def resize_patient(info: pd.Series):
    patient = info['name(raw)']
    try:
        data = loader({
            'origin': data_dir / patient / info['T2'],
            'output': output_dir / patient / 'AT.nii.gz',
        })
    except:
        return
    resizer = monai.transforms.Compose([
        monai.transforms.SpatialCropD('origin', roi_slices=[slice(None), slice(None), slice(0, 16)]),
        monai.transforms.ResizeD('output', spatial_size=[*data['origin'].shape[1:-1], -1], mode='nearest'),
    ])
    data = resizer(data)
    saver = monai.transforms.SaveImageD(
        ['origin', 'output'],
        output_dir=data_dir / patient,
        resample=False,
        separate_folder=False,
        print_log=True,
    )
    saver(data)

def main():
    cohort_info = read_cohort_info()
    # process_map(resize_patient, [info for _, info in cohort_info.iterrows()])
    for _, info in cohort_info.iterrows():
        resize_patient(info)

if __name__ == '__main__':
    main()
