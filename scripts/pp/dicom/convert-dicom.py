import os
from pathlib import Path

from luolib.utils import process_map

data_dir = Path('MB-data/new-test')
DICOM_dir = data_dir / 'DICOM'
NIfTI_dir = data_dir / 'NIfTI'
NIfTI_dir.mkdir()

def main():
    commands = []
    for study_dir in DICOM_dir.iterdir():
        # https://youtrack.jetbrains.com/issue/PY-78044/pycharm-cant-infer-type-from-pathlib.Path....iterdir
        study_dir: Path
        if not study_dir.is_dir():
            continue
        save_dir = NIfTI_dir / study_dir.name
        save_dir.mkdir()
        for series_dir in study_dir.iterdir():
            series_dir: Path
            if not series_dir.is_dir():
                continue
            cmd = f'dcm2niix -o {save_dir} -f %f-%p {series_dir}'
            commands.append(cmd)
    process_map(os.system, commands, max_workers=16)

if __name__ == '__main__':
    main()
