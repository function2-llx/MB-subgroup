import json
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm

data_dir = Path('MB-data/new-test')
src_dir = data_dir / 'NIfTI'
dst_dir = data_dir / 'NIfTI-axial'
dst_dir.mkdir(exist_ok=True)


def is_axial(image_file: Path, tol: float = 1e-2) -> bool:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(image_file))
    reader.ReadImageInformation()
    direction = reader.GetDirection()
    return 1 - abs(direction[0]) < tol and 1 - abs(direction[4]) < tol

def main():
    for study_dir in tqdm(list(src_dir.iterdir()), dynamic_ncols=True):
        if not study_dir.is_dir():
            continue
        for meta_file in study_dir.glob('*.json'):
            # 检查是否有超过2个同名文件
            if len(list(study_dir.glob(f'{meta_file.stem}.*'))) > 2:
                continue
            series_num = meta_file.stem.split('-', 1)[0]
            if len(image_files :=list(study_dir.glob(f'{series_num}-*.nii'))) > 1:
                continue
            image_file = image_files[0]
            if not is_axial(image_file):
                continue
            meta = json.loads(meta_file.read_bytes())
            dst_study_dir = dst_dir / study_dir.name
            dst_study_dir.mkdir(exist_ok=True)
            (dst_study_dir / image_file.name).hardlink_to(image_file)
            (dst_study_dir / meta_file.name).hardlink_to(meta_file)

if __name__ == '__main__':
    main()
