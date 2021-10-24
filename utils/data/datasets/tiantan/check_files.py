import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.axes import Axes
from monai.transforms import LoadImage

from utils.dicom_utils import ScanProtocol

loader = LoadImage()

results = {}

def check_patient(patient_dir: Path):
    patient = patient_dir.name
    if patient in results:
        return

    files = list(filter(lambda s: not s.name.startswith('.'), patient_dir.glob('*.nii.gz')))

    def find_file(pattern: str) -> Optional[str]:
        pattern = pattern.lower()
        for file in files:
            if file.name.lower().endswith(f'{pattern}.nii.gz'):
                return file.name
        return None

    result = {}
    for protocol in ScanProtocol:
        result[protocol.name] = find_file(protocol.name)
    for seg in ['AT', 'CT', 'WT']:
        result[seg] = find_file(seg)

    results[patient] = result
    if sum(v is not None for v in result.values()) < 5:
        print(patient, list(map(lambda file: file.name, files)))

def main():
    data_dir = Path('MB-MRI(2021-10-07)')
    try:
        for patient in data_dir.iterdir():
            check_patient(patient)
    finally:
        with open('files.json', 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        pd.DataFrame(results).transpose().to_excel('results.xlsx')

if __name__ == '__main__':
    main()
