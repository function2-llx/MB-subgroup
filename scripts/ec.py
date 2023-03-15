from pathlib import Path

import pandas as pd
import pydicom

def main():
    results = []
    for patient_dir in Path().iterdir():
        if not patient_dir.is_dir():
            continue
        sex = None
        age = None
        for sample_dcm_path in patient_dir.rglob('*.dcm'):
            ds = pydicom.dcmread(sample_dcm_path)
            if hasattr(ds, 'PatientSex'):
                sex = ds.PatientSex
            if hasattr(ds, 'PatientAge'):
                age = ds.PatientAge
            if sex is not None and age is not None:
                break

        results.append({
            'name': patient_dir.name,
            'sex': sex,
            'age': age,
        })

    pd.DataFrame.from_records(results).to_excel('clinical.xlsx', index=False)

if __name__ == '__main__':
    main()
