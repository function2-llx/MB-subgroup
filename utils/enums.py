from enum import Enum, unique

@unique
class Plane(Enum):
    Sagittal = 0,
    Coronal = 1,
    Axial = 2

@unique
class Protocol(Enum):
    T1 = 0,
    T1c = 1,
    T2 = 2

