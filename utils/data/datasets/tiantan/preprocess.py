"""
    crop and fix the affine
"""

from collections.abc import Mapping, Hashable
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
import monai.transforms
from monai.config import KeysCollection
from monai.transforms.spatial.dictionary import InterpolateModeSequence
from monai.utils import InterpolateMode

from utils.dicom_utils import ScanProtocol
from utils.transforms import SquareWithPadOrCropD

output_dir = Path('preprocessed')

seg_ref = {
    'AT': ScanProtocol.T2,
    'CT': ScanProtocol.T1c,
}

img_keys = list(ScanProtocol)
seg_keys = list(seg_ref)
all_keys = img_keys + seg_keys

class AlignShapeD(monai.transforms.MapTransform):
    def __init__(self, keys: KeysCollection, ref_key: str, mode: InterpolateModeSequence, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.ref_key = ref_key
        self.mode = mode

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        resizer = monai.transforms.ResizeD(self.keys, spatial_size=data[self.ref_key].shape[1:], mode=self.mode)
        return resizer(data)

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(all_keys),
    monai.transforms.AddChannelD(all_keys),
    monai.transforms.OrientationD(all_keys, 'LAS'),
    AlignShapeD(img_keys, ScanProtocol.T2, InterpolateMode.AREA),
    AlignShapeD(seg_keys, ScanProtocol.T2, InterpolateMode.NEAREST),
    monai.transforms.CropForegroundD(all_keys, source_key=ScanProtocol.T2),
    SquareWithPadOrCropD(all_keys),
])

def process_patient(info):
    patient = info['name(raw)']
    data = loader({
        **{
            protocol: Path('stripped') / patient / info[protocol]
            for protocol in img_keys
        },
        **{
            seg_type: Path('origin') / patient / info[seg_type]
            for seg_type in seg_keys
        }
    })
    for seg_type, img_ref in seg_ref.items():
        data[f'{seg_type}_meta_dict']['affine'] = data[f'{img_ref.name}_meta_dict']['affine']
    saver = monai.transforms.SaveImageD(
        all_keys,
        resample=False,
        output_dir=output_dir / patient,
        output_postfix='',
        separate_folder=False,
        print_log=False,
    )
    saver(data)

def main():
    cohort = pd.read_excel('cohort.xlsx')
    process_map(process_patient, [info for _, info in cohort.iterrows()])

if __name__ == '__main__':
    main()
