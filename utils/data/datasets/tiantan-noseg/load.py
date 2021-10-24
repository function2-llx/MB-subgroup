import json
from copy import deepcopy
from typing import Optional, Dict, List

import monai.transforms as monai_transforms
from tqdm.contrib.concurrent import process_map

from utils.data.datasets.tiantan import dataset_dir
from utils.data.datasets.tiantan.preprocess import output_dir as data_dir
from utils.dicom_utils import ScanProtocol

loader: Optional[monai_transforms.Compose] = None
# map label string to int
target_dict: Dict[str, int]

data_dir = dataset_dir / data_dir

def load_info(info):
    label = target_dict[info['subgroup']]
    patient = info['patient']
    scans = {
        protocol: str(dataset_dir / data_dir / patient / f'{protocol.name}.nii.gz')
        for protocol in ScanProtocol
    }
    data = loader(scans)
    data['label'] = label
    data['sex'] = info['sex']
    data['age'] = info['age']
    data['weight'] = info['weight']
    data['patient'] = patient
    return data

cohort: Optional[List[Dict]] = None

# return the cohort grouped by fold, return the flattened cohort if `n_folds` is None
def load_cohort(args, n_folds=None):
    global loader, target_dict, cohort

    assert len(args.protocols) == 3

    if cohort is None:
        target_dict = args.target_dict
        loader = monai_transforms.Compose([
            monai_transforms.LoadImageD(args.protocols),
            monai_transforms.AddChannelD(args.protocols),
            monai_transforms.ConcatItemsD(args.protocols, 'img'),
            monai_transforms.SelectItemsD('img'),
            monai_transforms.ThresholdIntensityD('img', threshold=0),
            monai_transforms.NormalizeIntensityD('img', channel_wise=True, nonzero=True),
        ])
        cohort = filter(lambda info: info['subgroup'] in target_dict, json.load(open(dataset_dir / 'cohort.json')))
        cohort = process_map(load_info, list(cohort), desc='loading cohort', ncols=80)

    if n_folds is not None:
        folds = json.load(open(dataset_dir / f'folds-{n_folds}.json'))
        folds = list(map(set, folds))
        return [[sample for sample in cohort if sample['patient'] in fold] for fold in folds]
    else:
        return deepcopy(cohort)
