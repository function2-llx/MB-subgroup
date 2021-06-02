__all__ = ['parser', 'process_args']

from argparse import ArgumentParser

from pathlib import Path

parser = ArgumentParser(add_help=False)

parser.add_argument(
    '--model',
    default='resnet',
    choices=['unet', 'resnet', 'resnet2p1d', 'preresnet', 'wideresnet', 'resnext', 'densenet', 'medicalnet'],
)
parser.add_argument('--model_depth', type=int, choices=[10, 18, 34, 50, 101], default=18)
parser.add_argument('--resnet_shortcut', default='B', choices=['A', 'B'], help='Shortcut type of resnet')
parser.add_argument('--n_input_channels', type=int, default=3)
parser.add_argument('--sample_size', default=224, type=int, help='Height and width of inputs')
parser.add_argument('--sample_slices', default=32, type=int, help='slices of inputs, temporal size in terms of videos')
parser.add_argument('--pretrain_root', default='pretrained', type=Path, help='root directory for pretrained models')
parser.add_argument('--pretrain_name', default=None, type=Path, help='Pretrained model name, also save path related to pretrain_root')
parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
parser.add_argument('--random_center', action='store_true', help='random center crop for augmentation')
parser.add_argument(
    '--resnet_widen_factor',
    default=1.0,
    type=float,
    help='The number of feature maps of resnet is multiplied by this value'
)
parser.add_argument('--conv1_t_size', default=7, type=int, help='Kernel size in t dim of conv1.')
parser.add_argument('--conv1_t_stride', default=1, type=int, help='Stride in t dim of conv1.')
parser.add_argument('--no_max_pool', action='store_true', help='If true, the max pooling after conv1 is removed.')
parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')

def process_args(args):
    if args.pretrain_name == 'scratch':
        args.pretrain_name = None
    return args
