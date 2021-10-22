import json
from pathlib import Path

import pydicom
import pandas as pd

if __name__ == '__main__':
    rows = []
    for patient_dir in Path('nifti').iterdir():
        if not patient_dir.is_dir():
            continue
        # patient_dir = next(patient_dir.iterdir())
        dcm_patient_dir = next((Path('dcm') / patient_dir.name).iterdir())
        try:
            sample_slice_path = next(dcm_patient_dir.rglob('*.dcm'))
        except StopIteration:
            continue
        sample_ds = pydicom.dcmread(sample_slice_path)
        # print(sample_ds, file=open('test.txt', 'w'))
        scan_dirs = {
            int(str(scan_dir.name).split('_')[0]): Path(*scan_dir.parts[2:])
            for scan_dir in dcm_patient_dir.iterdir()
        }
        row = [patient_dir.name, sample_ds.PatientName]
        for protocol in ['T1', 'T1c', 'T2']:
            scan_info = patient_dir / f'{protocol}.json'
            if scan_info.exists():
                scan_info = json.load(open(scan_info))
                series_number = scan_info['SeriesNumber']
                scan_dir = scan_dirs[series_number]
                row.append(scan_dir)
            else:
                row.append('')
        rows.append(row)
    pd.DataFrame(rows, columns=['编号', 'Name', 'T1', 'T1c', 'T2']).to_excel('protocols.xlsx', index=False)
