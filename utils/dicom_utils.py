# from enum import unique, Enum, auto, EnumMeta
from typing import Optional

import numpy as np
from aenum import Enum, unique, auto, EnumMeta

@unique
class Plane(Enum):
    Sagittal = 0
    Coronal = 1
    Axial = 2

# class ScanProtocolMeta(EnumMeta):
#     def __call__(cls, value):
#         if isinstance(value, str):
#             return cls.protocol_map[value.lower()]
#         return super().__call__(value)

class ScanProtocol(Enum):
    T1 = auto()
    T1c = auto()
    T2 = auto()

# ScanProtocolMeta.protocol_map = {
#     protocol.name.lower(): protocol
#     for protocol in ScanProtocol
# }

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
