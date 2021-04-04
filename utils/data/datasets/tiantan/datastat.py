from argparse import Namespace
from pathlib import Path

import numpy as np
from monai.transforms import LoadImaged
from tqdm import tqdm

from utils.data.datasets.tiantan import load_all
from utils.dicom_utils import ScanProtocol

if __name__ == '__main__':
    shapes_path = Path('shapes.npy')
    if shapes_path.exists():
        shapes = np.load(shapes_path)
    else:
        shapes = []
        for data in tqdm(
            load_all(
                Namespace(target_dict={name: i for i, name in enumerate(['WNT', 'SHH', 'G3', 'G4'])}, debug=False),
                loader=LoadImaged(ScanProtocol),
            ),
            ncols=80,
        ):
            for key in ScanProtocol:
                img = data[key]
                shapes.append(img.shape[:2])
        shapes = np.array(shapes)
        np.save(shapes_path, shapes)
    print('shape mean: ', shapes.mean(axis=0))
    print('shape std', shapes.std(axis=0))
    print('shape median', np.median(shapes, axis=0))
