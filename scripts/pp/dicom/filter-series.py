import json
import re
from pathlib import Path

import cytoolz
import numpy as np
from tqdm import tqdm

data_dir = Path('MB-data/new-test')
src_dir = data_dir / 'NIfTI'
dst_dir = data_dir / 'NIfTI-filtered'
dst_dir.mkdir(exist_ok=True)

def is_axial_orient(slice_orient: tuple[float, ...], tol: float = 1e-2) -> bool:
    direction = np.cross(slice_orient[:3], slice_orient[3:]).astype(np.float64)
    direction /= np.linalg.norm(direction)
    return 1 - abs(direction[2]) < tol

def check_modality(meta: dict) -> bool:
    series_desc: str = meta['SeriesDescription']
    series_desc = series_desc.upper()
    return 'T1' in series_desc or 'T2' in series_desc

def check_meta(meta: dict):
    if (slice_orient := meta.get('ImageOrientationPatientDICOM')) is None or not is_axial_orient(slice_orient):
        return False
    return check_modality(meta)

def main():
    for study_dir in tqdm(list(src_dir.iterdir()), dynamic_ncols=True):
        if not study_dir.is_dir():
            continue
        for meta_file in study_dir.glob('*.json'):
            try:
                series_num = int(meta_file.stem)
            except ValueError:
                continue
            # make sure there are only $sn.json and $sn.nii for this series
            pattern = re.compile(rf'{series_num}\D?.*')
            if cytoolz.count(
                filter(
                    lambda path: pattern.match(str(path)) is not None,
                    study_dir.glob(f'{series_num}*'),
                ),
            ) > 2:
                continue
            meta = json.loads(meta_file.read_bytes())
            assert series_num == meta['SeriesNumber']
            if not check_meta(meta):
                continue
            image_file = study_dir / f'{series_num}.nii'
            dst_study_dir = dst_dir / study_dir.name
            dst_study_dir.mkdir(exist_ok=True)
            (dst_study_dir / image_file.name).hardlink_to(image_file)
            (dst_study_dir / meta_file.name).hardlink_to(meta_file)

if __name__ == '__main__':
    main()
