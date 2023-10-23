import itertools as it
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from luolib.nnunet import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.utils import (
    create_lists_from_splitted_dataset_folder,
    get_identifiers_from_splitted_dataset_folder,
)

dataset_id = 500
dataset_name = convert_id_to_dataset_name(dataset_id)

def get_filenames_of_train_images_and_targets(raw_dataset_folder: Path, dataset_json: dict):
    identifiers = get_identifiers_from_splitted_dataset_folder(str(raw_dataset_folder / 'imagesTs'), dataset_json['file_ending'])
    images = create_lists_from_splitted_dataset_folder(
        str(raw_dataset_folder / 'imagesTs'),
        dataset_json['file_ending'],
        identifiers,
    )
    segs = [
        str(seg_path) if (seg_path := raw_dataset_folder / 'labelsTs' / (i + dataset_json['file_ending'])).exists()
        else None
        for i in identifiers
    ]
    dataset = {i: {'images': im, 'label': se} for i, im, se in zip(identifiers, images, segs)}
    return dataset

def run_case(
    preprocessor: DefaultPreprocessor,
    output_filename_truncated: str,
    image_files: list[str],
    seg_file: str | None,
    *args
):
    data, seg, data_properties = preprocessor.run_case(image_files, seg_file, *args)
    np.save(output_filename_truncated + '.npy', data)
    if seg_file is not None:
        np.save(output_filename_truncated + '_seg.npy', seg)
    pd.to_pickle(data_properties, output_filename_truncated + '.pkl')

def main():
    plans_file = Path(nnUNet_preprocessed) / dataset_name / 'nnUNetPlans.json'
    plans_manager = PlansManager(plans_file)
    dataset_json = json.loads((nnUNet_preprocessed / dataset_name / 'dataset.json').read_bytes())
    configuration_manager = plans_manager.get_configuration('3d_fullres')
    preprocessor = configuration_manager.preprocessor_class(verbose=True)
    output_directory = nnUNet_preprocessed / dataset_name / configuration_manager.data_identifier
    dataset = get_filenames_of_train_images_and_targets(nnUNet_raw / dataset_name, dataset_json)
    process_map(
        run_case,
        it.repeat(preprocessor),
        *zip(*(
            (str(output_directory / k), dataset[k]['images'], dataset[k]['label'])
            for k in dataset
        )),
        it.repeat(plans_manager),
        it.repeat(configuration_manager),
        it.repeat(dataset_json),
        max_workers=16,
        ncols=80,
        total=len(dataset),
    )

if __name__ == '__main__':
    main()
