from argparse import Namespace
from pathlib import Path
import json

import numpy as np
from monai import transforms as monai_transforms
from tqdm import tqdm

from utils.data.datasets.tiantan import load_all
from utils.dicom_utils import ScanProtocol
from preprocess.convert import output_dir

loader = monai_transforms.Compose([
    monai_transforms.LoadImaged('img'),
    monai_transforms.AddChanneld('img'),
    # monai_transforms.Orientationd('img', axcodes='LAS'),
    # monai_transforms.SpacingD('img', spacing)
    # monai_transforms.ThresholdIntensityd('img', threshold=0),
])




if __name__ == '__main__':
    cohort = json.load(open('cohort.json'))
    for info in cohort:
        for scan in info['scans']:
            scan_info = json.load(open(output_dir / info['patient'] / f'{scan}.json'))
            scan_path = output_dir / info['patient'] / f'{scan}_ss.nii.gz'
            print(scan_path)
            img = loader({'img': str(scan_path)})
            # print(scan_info['SpacingBetweenSlices'])
            print('before:', img['img'].shape[1:], img['img_meta_dict']['pixdim'])
            spacing = monai_transforms.SpacingD('img', pixdim=(1, 1, img['img_meta_dict']['pixdim'][3]))
            img = spacing(img)
            print('after:', img['img'].shape[1:], img['img_meta_dict']['pixdim'])
            # resample_path = output_dir / info['patient'] / f'{scan}_re.nii.gz'
            # print(img[['original_affine'])
            # del img['img_meta_dict']['original_affine']
            # monai_transforms.SaveImageD('img')(img)
            # exit(0)
