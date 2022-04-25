from pathlib import Path

import pandas as pd

from mbs.utils.dicom_utils import ScanProtocol

data_dir = Path('stripped')

def main():
    cohort = pd.read_excel('cohort.xlsx', index_col='name(raw)')
    for patient, info in cohort.iterrows():
        patient_dir = data_dir / patient
        for protocol in ScanProtocol:
            protocol = protocol.name
            (patient_dir / f'{protocol}.nii.gz').rename(patient_dir / info[protocol])

if __name__ == '__main__':
    main()
