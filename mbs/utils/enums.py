from pathlib import Path

from monai.utils import StrEnum

class MBDataKey(StrEnum):
    NUMBER = 'number'
    GROUP = 'group'
    SUBGROUP = 'subgroup'
    CLINICAL = 'clinical'

class MBGroup(StrEnum):
    CHILD = 'child'
    ADULT = 'adult'

class Modality(StrEnum):
    T1 = 'T1'
    T1C = 'T1C'
    T2 = 'T2'

class SegClass(StrEnum):
    ST = 'ST'
    AT = 'AT'
    CT = 'CT'

SUBGROUPS = ['WNT', 'SHH', 'G3', 'G4']
DATASET_ROOT = Path(__file__).parents[1]
DATA_DIR = DATASET_ROOT / 'origin'
PROCESSED_DIR = DATASET_ROOT / 'processed'
CLINICAL_DIR = DATA_DIR / 'clinical'
SEG_REF = {
    SegClass.AT: Modality.T2,
    SegClass.CT: Modality.T1C,
    SegClass.ST: Modality.T2,
}
