from pathlib import Path

import monai
import numpy as np
from monai.data import write_nifti
from tqdm.contrib.concurrent import process_map

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD('img', series_meta=True),
    monai.transforms.AddChannelD('img'),
    monai.transforms.OrientationD('img', 'RAS'),
])

data_dir = Path('patients/Image of medulloblastoma - remarkable/pre-opeMRI')
output_dir = Path('transformed')

def process_scan(save_dir: Path, scan_dir: Path, scan_id: int):
    try:
        data = loader({'img': str(scan_dir)})
    except Exception as e:
        return
    spacing = np.linalg.norm(data['img_meta_dict']['affine'][:3, :3], axis=0)
    write_nifti(
        data=np.moveaxis(np.asarray(data['img']), 0, -1),
        file_name=str(save_dir / f'{scan_id}: {data["img_meta_dict"]["0008|103e"].strip()}.nii.gz'),
        affine=np.diag(np.append(spacing, 1)),
    )

def main():
    all_scans = []
    for patient_dir in data_dir.iterdir():
        if not patient_dir.is_dir():
            continue
        for study_dir in patient_dir.iterdir():
            try:
                next(study_dir.glob('*/*.dcm'))
            except StopIteration:
                continue
            break
        else:
            continue

        save_dir = output_dir / patient_dir.name
        save_dir.mkdir(exist_ok=True, parents=True)
        for scan_dir in study_dir.iterdir():
            if not scan_dir.is_dir():
                continue
            scan_id = int(scan_dir.name.split("_")[0])
            if scan_id < 20:
                all_scans.append((save_dir, scan_dir, scan_id))

    process_map(process_scan, *zip(*all_scans), max_workers=1, ncols=80)

if __name__ == '__main__':
    main()
