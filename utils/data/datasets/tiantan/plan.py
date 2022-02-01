from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, List, Union, Optional

import numpy as np

import monai
from tqdm.contrib.concurrent import process_map

from cls_unet.cls_unet import get_params
from utils.data.datasets.tiantan.load import read_cohort_info

data_dir = Path('preprocessed')
loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(['T2', 'AT']),
])

def process_patient(_, info):
    patient_dir = data_dir / info['name(raw)']
    data = loader({'T2': patient_dir / info['T2'], 'AT': patient_dir / info['AT']})
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
    kernels, strides, bn_size = get_params(tuple(size), tuple(spacing), min_fmap=2)
    print('spacing =', spacing)
    print('size =', size)
    print('kernels =', kernels)
    print('strides =', strides)
    print('depth =', len(kernels) - 1)
    print('bottleneck size =', bn_size)

if __name__ == '__main__':
    main()
