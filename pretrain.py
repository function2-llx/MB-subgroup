from pathlib import Path

from utils.dicom_utils import ScanProtocol

def parse_args():
    from argparse import ArgumentParser
    import utils.args
    import resnet_3d.model

    parser = ArgumentParser(parents=[utils.args.parser, resnet_3d.model.parser])
    protocol_names = [value.name for value in ScanProtocol]
    parser.add_argument('--protocols', nargs='+', choices=protocol_names, default=protocol_names)
    parser.add_argument('--output_root', type=Path, required=True)

    args = parser.parse_args()
    args.protocols = list(map(ScanProtocol.__getitem__, args.protocols))

    return args

if __name__ == '__main__':
