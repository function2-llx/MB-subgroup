from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from utils.dicom_utils import ScanProtocol

@dataclass
class MBArgs:
    img_dir: Path = field(default=None)
    seg_dir: Path = field(default=None)
    segs: List[str] = field(default_factory=list)
    subgroups: List[str] = field(default=None)

    def __post_init__(self):
        protocol_map = {
            protocol.name.lower(): protocol
            for protocol in list(ScanProtocol)
        }
        self.protocols = list(map(lambda name: protocol_map[name.lower()], self.protocols))
        self.img_dir = Path(self.img_dir)
        self.seg_dir = Path(self.seg_dir)
