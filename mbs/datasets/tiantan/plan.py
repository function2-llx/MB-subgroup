from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

import monai

from cls_unet.cls_unet import get_params
from mbs.datasets.tiantan.load import read_cohort_info

data_dir = Path('preprocessed')
loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(['T2', 'AT']),
])

def process_patient(_, info):
    patient_dir = data_dir / info['subject']
    data = loader({'T2': patient_dir / 'T2.nii.gz', 'AT': patient_dir / 'AT.nii.gz'})
    spacing = data['T2_meta_dict']['pixdim'][1:4]
    size = data['T2'].shape
    pos = data['AT'].sum(axis=(0, 1)).nonzero()[0][-1]
    return spacing, size, pos

sample_slices = 16

def main():
    cohort_info = read_cohort_info()
    spacings, sizes, pos = map(np.array, zip(*process_map(process_patient, *zip(*cohort_info.iterrows()))))
    print(pos)

    spacing = np.median(spacings, axis=0)
    size = list(map(int, np.median(sizes, axis=0)))
    size[2] = sample_slices
    params = get_params(tuple(size), tuple(spacing), min_fmap=2)
    print('spacing =', spacing)
    print('size =', size)
    print('kernels =', params.kernels)
    print('strides =', params.strides)
    print('depth =', len(params.kernels) - 1)
    print('bottleneck size =', params.fmap_shapes[-1])

if __name__ == '__main__':
    main()
