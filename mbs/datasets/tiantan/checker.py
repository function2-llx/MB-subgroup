import itertools
from sys import argv
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
import monai

seg_keys = ['AT', 'CT', 'ST']
all_keys = ['T1C', 'T2'] + seg_keys

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(all_keys),
    monai.transforms.AddChannelD(all_keys),
])

# 在这里改成默认检查路径，注意不要用反斜杠
check_dir = Path('test')
suffix = '.nii'

def check(check_dir: Path, patient: str):
    data = loader({
        key: check_dir / patient / f'{key}{suffix}'
        for key in all_keys
    })
    t1c_shape = data['T1C'].shape
    t2_shape = data['T2'].shape
    ret = {
        'patient': patient,
        'T1C shape': ' '.join(map(str, t1c_shape[1:])),
        'T2 shape': ' '.join(map(str, t2_shape[1:])),
    }
    for key in seg_keys:
        x = data[key]
        ref_shape = t1c_shape if key == 'CT' else t2_shape
        if x.shape != ref_shape:
            ret[f'{key} shape'] = ' '.join(map(str, x.shape[1:]))
        values = np.unique(x)
        if not np.array_equal(values, [0, 1]):
            ret[f'{key} values'] = values
    return ret

def main():
    global check_dir
    if len(argv) > 1:
        check_dir = Path(argv[1])
    print(check_dir)
    cohort = [patient_path.name for patient_path in check_dir.iterdir()]
    print('cohort size:', len(cohort))
    pd.DataFrame.from_records(
        process_map(check, itertools.repeat(check_dir, len(cohort)), cohort, ncols=80)
    ).set_index('patient').to_excel('check-results.xlsx')

if __name__ == '__main__':
    main()
