from collections import ChainMap
from pathlib import Path

import monai
from monai.utils import ImageMetaKey, InterpolateMode
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data
from tqdm.contrib.concurrent import process_map

from mbs.utils.argparse import ArgParser
from mbs.datasets.tiantan.args import MBArgs
from mbs.datasets.tiantan.load import read_cohort_info
from mbs.utils.dicom_utils import ScanProtocol

args, = ArgParser([MBArgs], use_conf=False).parse_args_into_dataclasses()
args: MBArgs

task_id = '500'
# Tiantan MB
task_name = 'TTMB'
target_base = Path(nnUNet_raw_data) / f'Task{task_id}_{task_name}'
target_base.mkdir(parents=True, exist_ok=True)
target_images_train = target_base / 'imagesTr'
target_labels_train = target_base / 'labelsTr'

for path in [target_images_train, target_labels_train]:
    path.mkdir(exist_ok=True)

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(list(ScanProtocol) + ['AT']),
    monai.transforms.AddChannelD(list(ScanProtocol) + ['AT']),
    monai.transforms.OrientationD(list(ScanProtocol) + ['AT'], 'LAS'),
    monai.transforms.SpatialCropD(list(ScanProtocol) + ['AT'], roi_slices=[slice(None), slice(None), slice(0, 16)]),
    monai.transforms.ThresholdIntensityD('AT', threshold=1, above=False, cval=1),
    monai.transforms.CastToTypeD('AT', int),
    monai.transforms.ResizeD(list(ScanProtocol), spatial_size=(448, 448, -1)),
    monai.transforms.ResizeD(
        'AT',
        spatial_size=(448, 448, -1),
        mode=InterpolateMode.NEAREST,
    ),
])

def convert(info):
    patient = info['name(raw)']
    data = loader({
        **{
            protocol: args.img_dir / patient / info[protocol.name]
            for protocol in list(ScanProtocol)
        },
        'AT': args.seg_dir / patient / info['AT'],
    })
    saver = monai.transforms.Compose([
        monai.transforms.Lambda(lambda data: {
            **ChainMap(*(
                {
                    protocol: data[protocol],
                    f'{protocol}_meta_dict': {
                        **data[f'{protocol}_meta_dict'],
                        ImageMetaKey.FILENAME_OR_OBJ: f'{patient}_{i:04d}'
                    }
                }
                for i, protocol in enumerate(ScanProtocol)
            )),
            **{
                'AT': data['AT'],
                'AT_meta_dict': {
                    **data['AT_meta_dict'],
                    ImageMetaKey.FILENAME_OR_OBJ: patient,
                }
            }
        }),
        monai.transforms.SaveImageD(
            list(ScanProtocol),
            output_dir=target_images_train,
            separate_folder=False,
            resample=False,
            output_postfix='',
        ),
        monai.transforms.SaveImageD(
            'AT',
            output_dir=target_labels_train,
            separate_folder=False,
            resample=False,
            output_postfix='',
        )
    ])
    saver(data)

def main():
    cohort = read_cohort_info(args)
    process_map(convert, [info for _, info in cohort.iterrows()])
    generate_dataset_json(
        output_file=str(target_base / 'dataset.json'),
        imagesTr_dir=str(target_images_train),
        imagesTs_dir=None,
        modalities=tuple(protocol.name for protocol in list(ScanProtocol)),
        labels={0: 'background', 1: 'AT'},
        dataset_name=task_name,
    )

if __name__ == '__main__':
    main()
