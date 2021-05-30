import json
from pathlib import Path

from convert import output_dir
src = Path('processed-tmp')

if __name__ == '__main__':
    cohort = json.load(open('cohort.json'))
    for patient in cohort:
        for scan in patient['scans']:
            for post in ['', '_mask']:
                target = Path(patient['patient']) / f'{scan}_ss{post}.nii.gz'
                path = src / target
                assert path.exists()
                path.rename(output_dir / target)
