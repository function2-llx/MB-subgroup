import logging

import torch
from torch import nn

from args import parser
from .models.resnet import ResNet

parser.add_argument('--model_depth', type=int, choices=[10, 18, 34, 50])
parser.add_argument('--use8', action='store_true', help='use model pre-trained on 8 datasets')
parser.add_argument(
    '--n_seg_classes',
    default=2,
    type=int,
    help="Number of segmentation classes",
)

def generate_model(args, pretrain: bool = True) -> ResNet:
    from .models import resnet

    assert args.model_depth in [10, 18, 34, 50, 101, 152, 200]

    kargs = {
        'shortcut_type': args.resnet_shortcut,
        'num_classes': len(args.target_names),
        'num_seg_classes': args.n_seg_classes,
    }

    if args.model_depth == 10:
        model = resnet.resnet10(**kargs)
    elif args.model_depth == 18:
        model = resnet.resnet18(**kargs)
    elif args.model_depth == 34:
        model = resnet.resnet34(**kargs)
    elif args.model_depth == 50:
        model = resnet.resnet50(**kargs)
    elif args.model_depth == 101:
        model = resnet.resnet101(**kargs)
    elif args.model_depth == 152:
        model = resnet.resnet152(**kargs)
    elif args.model_depth == 200:
        model = resnet.resnet200(**kargs)
    else:
        raise ValueError('unreachable')

    if pretrain:
        # pretrain loading wrapper
        model = nn.DataParallel(model, device_ids=None)
        if args.rank == 0:
            logging.info(f'load pre-trained weights from {args.pretrain_path}')
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.pretrain_path)['state_dict'], strict=False)
        assert len(unexpected_keys) == 0
        allowed_missing_keys = ['conv_seg', 'fc']
        for missing_key in missing_keys:
            assert any(allow in missing_key for allow in allowed_missing_keys)
        model = model.module

    return model.to(args.device)
