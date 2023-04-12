from monai.utils import StrEnum

class MBDataKey(StrEnum):
    CASE = 'case'
    SUBGROUP_ID = 'subgroup_id'

class MBGroup(StrEnum):
    CHILD = 'child'
    ADULT = 'adult'

class Modality(StrEnum):
    T1 = 'T1'
    T1C = 'T1C'
    T2 = 'T2'

class SegClass(StrEnum):
    ST = 'ST'
    CT = 'CT'
    AT = 'AT'

SUBGROUPS = ['WNT', 'SHH', 'G3', 'G4']
