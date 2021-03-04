from enum import unique, Enum
from typing import Tuple, Optional


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


def parse_series_desc(desc: str) -> Tuple[Optional[Plane], Optional[ScanProtocol]]:
    desc = desc.lower()
    if 't1' in desc:
        protocol = ScanProtocol.T1c if '+c' in desc else ScanProtocol.T1
    elif 't2' in desc:
        protocol = ScanProtocol.T2
    else:
        protocol = None

    if 'tra_' in desc or 'ax' in desc:
        plane = Plane.Axial
    elif 'sag' in desc:
        plane = Plane.Sagittal
    elif 'cor' in desc:
        plane = Plane.Coronal
    else:
        plane = None

    return plane, protocol