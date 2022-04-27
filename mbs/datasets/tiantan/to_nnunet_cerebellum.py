from pathlib import Path

from tqdm.contrib.concurrent import process_map

import monai
from monai.utils import ImageMetaKey
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data

from mbs.utils.dicom_utils import ScanProtocol


img_dir = Path('cerebellum-stripped')
seg_dir = Path('cerebellum-data')

task_id = '501'
task_name = 'TTMB-cerebellum'
target_base = Path(nnUNet_raw_data) / f'Task{task_id}_{task_name}'
target_base.mkdir(parents=True, exist_ok=True)
target_images_train = target_base / 'imagesTr'
target_labels_train = target_base / 'labelsTr'

label_key = 'ST'

for path in [target_images_train, target_labels_train]:
    path.mkdir(exist_ok=True)

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(list(ScanProtocol) + [label_key]),
    monai.transforms.AddChannelD(list(ScanProtocol) + [label_key]),
    monai.transforms.OrientationD(list(ScanProtocol) + [label_key], 'RAS'),
    monai.transforms.ThresholdIntensityD(label_key, threshold=1, above=False, cval=1),
    monai.transforms.CastToTypeD(label_key, int),
    # monai.transforms.ResizeD(list(ScanProtocol), spatial_size=(448, 448, -1)),
    # monai.transforms.ResizeD(
    #     label_key,
    #     spatial_size=(448, 448, -1),
    #     mode=InterpolateMode.NEAREST,
    # ),
])

saver = monai.transforms.Compose([
    monai.transforms.SaveImageD(
        protocol,
        output_dir=target_images_train,
        output_postfix=f'{i:04d}',
        resample=False,
        data_root_dir=str(img_dir),
        separate_folder=False,
    )
    for i, protocol in enumerate(ScanProtocol)
] + [
    monai.transforms.SaveImageD(
        label_key,
        output_dir=target_labels_train,
        output_postfix='',
        resample=False,
        data_root_dir=str(img_dir),
        separate_folder=False,
    )
])

def convert(patient: str):
    data = loader({
        **{
            protocol: img_dir / patient / f'{protocol.name.upper()}.nii.gz'
            for protocol in list(ScanProtocol)
        },
        label_key: seg_dir / patient / f'{label_key}.nii',
    })
    for key in list(ScanProtocol) + [label_key]:
        data[f'{key}_meta_dict'][ImageMetaKey.FILENAME_OR_OBJ] = patient
    saver(data)

def get_cohort() -> list[str]:
    ret = []
    for patient_dir in img_dir.iterdir():
        patient = patient_dir.name
        if all((patient_dir / f'{protocol}.nii').exists() for protocol in ScanProtocol) \
        and (seg_dir / patient / f'{label_key}.nii').exists():
            ret.append(patient_dir.name)
    return ret

def main():
    cohort = get_cohort()
    print(cohort, len(cohort))
    process_map(convert, cohort)
    generate_dataset_json(
        output_file=str(target_base / 'dataset.json'),
        imagesTr_dir=str(target_images_train),
        imagesTs_dir=None,
        modalities=tuple(protocol.name for protocol in list(ScanProtocol)),
        labels={0: 'background', 1: label_key},
        dataset_name=task_name,
    )

if __name__ == '__main__':
    main()
