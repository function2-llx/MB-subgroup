import logging
from dataclasses import dataclass
from pathlib import Path

import radiomics
from radiomics.featureextractor import RadiomicsFeatureExtractor
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from utils.args import ArgumentParser
from utils.data.datasets.tiantan.load import read_cohort_info
from utils.dicom_utils import ScanProtocol

@dataclass
class RadiomicsArgs:
    img_dir: Path = None
    seg_dir: Path = None
    output_dir: Path = None
    protocol: ScanProtocol = None
    seg: str = None

    def __post_init__(self):
        self.img_dir = Path(self.img_dir)
        self.seg_dir = Path(self.img_dir)
        self.output_dir = Path(self.output_dir)

args: RadiomicsArgs
extractor: RadiomicsFeatureExtractor

def extract_patient(info):
    name = info['name(raw)']
    # print(patient)
    features = extractor.execute(
        str(args.img_dir / name / info[args.protocol]),
        str(args.seg_dir / name / info[args.seg]),
        voxelBased=False,
    )
    features['name'] = name
    return features

def setup_logging():
    handler = logging.FileHandler(Path(args.output_dir) / 'extract.log', mode='w')
    radiomics.logger.addHandler(handler)
    radiomics.setVerbosity(logging.ERROR)

def main():
    global args, extractor
    parser = ArgumentParser([RadiomicsArgs])
    args, = parser.parse_args_into_dataclasses()
    setup_logging()
    extractor = RadiomicsFeatureExtractor('params/baseline.yml')

    radiomics.logger.info(extractor.enabledFeatures)
    cohort_info = read_cohort_info()
    all_features = process_map(extract_patient, [info for _, info in cohort_info.iterrows()], ncols=80, max_workers=16)
    # all_features = []
    # for _, info in cohort_info.iterrows():
    #     all_features.append(extract_patient(info))
    #     break
    all_features = pd.DataFrame.from_records(all_features).set_index('name')
    all_features.to_excel(args.output_dir / 'features.xlsx')

if __name__ == '__main__':
    main()
