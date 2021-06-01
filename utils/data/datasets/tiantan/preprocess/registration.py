import json
from pathlib import Path

from fsl.utils.platform import platform
from fsl.wrappers import flirt
from tqdm.contrib.concurrent import process_map

from histogram_matching import output_dir as input_dir
from utils.dicom_utils import ScanProtocol

output_dir = Path('sri24')
ref_dir = Path(platform.fsldir) / 'data' / 'sri24'

def process_patient(info):
    patient = info['patient']
    out_dir = Path(output_dir) / patient
    out_dir.mkdir(parents=True, exist_ok=True)
    for protocol in ScanProtocol:
        src = input_dir / patient / f'{protocol.name}.nii.gz'
        ref = ref_dir / ('late.nii' if protocol == ScanProtocol.T2 else 'spgr.nii')
        out = out_dir / f'{protocol.name}.nii.gz'
        print(src)
        flirt(str(src), str(ref), out=str(out))

if __name__ == '__main__':
    cohort = json.load(open('cohort.json'))
    process_map(process_patient, cohort, ncols=80)
