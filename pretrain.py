import logging
from pathlib import Path

from monai.transforms import *
from monai.networks.nets import UNet
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from torch import nn

from resnet_3d.model import generate_model
from resnet_3d.models.backbone import Backbone
from utils.data import MultimodalDataset
from utils.transforms import ToTensorDeviced


def parse_args():
    from argparse import ArgumentParser
    import utils.args
    import resnet_3d.model

    parser = ArgumentParser(parents=[utils.args.parser, resnet_3d.model.parser])
    parser.add_argument('--output_root', type=Path, default='pretrained')
    parser.add_argument('--datasets', choices=['brats20'], default=['brats20'])

    args = parser.parse_args()

    args.model_output_root = args.output_root \
        / '+'.join(args.datasets) \
        / 'ep{epochs},lr{lr},wd{weight_decay}'.format(**args.__dict__)

    return args

class PretrainModel(nn.Module):
    def __init__(self, backbone: Backbone):
        super().__init__()
        self.backbone = backbone
        pass

    def setup_logging(self):
        args = self.args
        handlers = [logging.StreamHandler()]
        args.model_output_root.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.model_output_root / 'train.log', mode='a'))
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt=logging.Formatter.default_time_format,
            level=logging.INFO,
            handlers=handlers
        )

    def forward(self, x):
        features = self.backbone.forward(x)


class Trainer:
    def __init__(self, args, dataset):
        self.args = args
        backbone = generate_model(args, pretrain=False)
        self.model = PretrainModel(backbone)

    def is_world_master(self) -> bool:
        return self.args.rank == 0

    def train(self):
        pass

if __name__ == '__main__':
    args = parse_args()
    data = []
    for dataset in args.datasets:
        import utils.data.datasets as datasets
        loader = {
            'brats20': datasets.brats20.load_all
        }[dataset]
        data.extend(loader(args))
        dataset = MultimodalDataset(data, Compose([
            ToTensorDeviced(('imgs'))
        ]))
