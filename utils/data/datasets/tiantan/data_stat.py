from pathlib import Path
from typing import Dict, Tuple

import monai
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from utils.data.datasets.tiantan.load import read_cohort_info
from utils.dicom_utils import ScanProtocol

data_dir = Path(__file__).parent / Path('cropped')
seg_ref = {
    'AT': ScanProtocol.T2,
    'CT': ScanProtocol.T1c,
}
img_keys = list(ScanProtocol)
seg_keys = list(seg_ref)
all_keys = img_keys + seg_keys

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(all_keys),
    monai.transforms.AddChannelD(all_keys),
    monai.transforms.OrientationD(all_keys, 'LAS'),
])

def stat_patient(info: pd.Series) -> Tuple[str, Dict]:
    patient = info['name(raw)']
    data = loader({
        key: data_dir / patient / info[key]
        for key in all_keys
    })
    ret = {}
    src_key = ScanProtocol.T2
    for i, dim in enumerate('HWD'):
        ret[dim] = data[src_key].shape[i + 1]
    for seg_key in seg_keys:
        seg_sum = np.sum(data[seg_key], axis=(0, 1, 2))
        last = 0
        while last < len(seg_sum) and seg_sum[last] == 0:
            last += 1
        while last < len(seg_sum) and seg_sum[last] > 0:
            last += 1
        ret[f'{seg_key}-last'] = last
    return patient, ret

def main():
    cohort = read_cohort_info()
    results = process_map(stat_patient, [info for _, info in cohort.iterrows()], ncols=80)
    # results = []
    # for _, info in cohort.iterrows():
    #     results.append(stat_patient(info))
    pd.DataFrame(dict(results)).transpose().to_excel('stat.xlsx')

if __name__ == '__main__':
    main()
