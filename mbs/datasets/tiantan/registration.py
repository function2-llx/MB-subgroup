from pathlib import Path

from fsl.wrappers import flirt
from tqdm.contrib.concurrent import process_map

from mbs.utils.dicom_utils import ScanProtocol

input_dir = Path('inter-rater-test/A')
output_dir = Path('inter-rater-test/A-registered')

def process_patient(patient: str):
    patient_input_dir = input_dir / patient
    patient_output_dir = output_dir / patient
    patient_output_dir.mkdir(parents=True, exist_ok=True)
    ref = patient_input_dir / f'{ScanProtocol.T2}.nii'
    for protocol in ScanProtocol:
        if protocol is not ScanProtocol.T2:
            continue
        src = patient_input_dir / f'{protocol}.nii'
        out = patient_output_dir / f'{protocol}.nii.gz'
        print(src)
        flirt(str(src), str(ref), out=str(out))

def get_cohort() -> list[str]:
    return [
        patient_dir.name
        for patient_dir in input_dir.iterdir()
    ]

if __name__ == '__main__':
    cohort = get_cohort()
    print(cohort)
    print(len(cohort))
    process_map(process_patient, cohort, ncols=80)
