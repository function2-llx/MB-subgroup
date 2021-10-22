from enum import unique, Enum
from typing import Optional

import numpy as np

@unique
class Plane(Enum):
    Sagittal = 0
    Coronal = 1
    Axial = 2

@unique
class ScanProtocol(Enum):
    T1 = 0
    T1c = 1
    T2 = 2

def get_plane(ds) -> Optional[Plane]:
    try:
        ortn = np.array(ds.ImageOrientationPatient)
    except AttributeError:
        return None
    return Plane(np.abs(np.cross(ortn[:3], ortn[3:])).argmax())

def parse_series_desc(desc: str) -> Optional[ScanProtocol]:
    desc = desc.lower()
    if 't1' in desc:
        protocol = ScanProtocol.T1c if '+c' in desc else ScanProtocol.T1
    elif 't2' in desc:
        protocol = ScanProtocol.T2
    else:
        protocol = None

    return protocol
