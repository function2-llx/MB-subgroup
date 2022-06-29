from pathlib import Path

from tqdm.contrib.concurrent import process_map
import numpy as np
import pandas as pd
import torch

import monai
from monai.metrics import DiceMetric

A_dir = Path('inter-rater-test/A')
B_dir = Path('inter-rater-test/B')
dice_metric = DiceMetric()

def get_cohort(patients_dir: Path) -> list[str]:
    return [patient_path.name for patient_path in patients_dir.iterdir()]

seg_keys = ['AT', 'CT', 'ST']
all_keys = ['T1C', 'T2'] + seg_keys

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(all_keys),
    monai.transforms.AddChannelD(all_keys),
    # monai.transforms.ToTensorD(keys),
])

def process(patient: str):
    a, b = map(loader, (
        {
            key: data_dir / patient / f'{key}.nii'
            for key in all_keys
        }
        for data_dir in [A_dir, B_dir]
    ))
    t1c_shape = a['T1C'].shape
    t2_shape = a['T2'].shape
    assert b['T1C'].shape == t1c_shape
    assert b['T2'].shape == t2_shape
    ret = {
        'patient': patient,
        'T1C shape': ' '.join(map(str, t1c_shape[1:])),
        'T2 shape': ' '.join(map(str, t2_shape[1:])),
    }
    for key in seg_keys:
        ref_shape = t1c_shape if key == 'CT' else t2_shape
        for rater in 'ab':
            x: np.ndarray = locals()[rater][key]
            ret[f'{rater}-{key}'] = ' '.join(map(str, np.unique(x)))
        ka = torch.tensor(a[key][np.newaxis])
        kb = torch.tensor(b[key][np.newaxis])
        mismatch = ''
        if ka.shape[1:] != ref_shape:
            mismatch += f'a ' + ' '.join(map(str, ka.shape[2:]))
        if kb.shape[1:] != ref_shape:
            if mismatch:
                mismatch += '\n'
            mismatch += f'b ' + ' '.join(map(str, kb.shape[2:]))
        if not mismatch:
            ret[f'{key} Dice'] = dice_metric(ka, kb).item()
        else:
            ret[f'{key} Dice'] = mismatch
    return ret

def main():
    cohort = list(set(get_cohort(A_dir)) & set(get_cohort(B_dir)))
    print('cohort size:', len(cohort))

    pd.DataFrame.from_records(
        process_map(process, cohort, ncols=80)
    ).set_index('patient').to_excel('inter-rater-test.xlsx')

if __name__ == '__main__':
    main()
