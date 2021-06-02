import logging
from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor, nn

from . import resnet, resnet2p1d, wide_resnet, resnext, pre_act_resnet, densenet
from .args import parser, process_args
from .backbone import Backbone
from .unet import UNet

def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]

def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})

    return parameters

def generate_model(opt, pretrain: bool = True, num_seg=None) -> Backbone:
    if opt.model == 'unet':
        model = UNet(
            dimensions=3,
            in_channels=3,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            n_classes=opt.n_classes,
        )
    elif opt.model == 'resnet':
        model = resnet.generate_model(
            model_depth=opt.model_depth,
            n_classes=opt.n_classes,
            n_input_channels=opt.n_input_channels,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool,
            widen_factor=opt.resnet_widen_factor,
            num_seg=num_seg,
        )
    elif opt.model == 'resnet2p1d':
        model = resnet2p1d.generate_model(
            model_depth=opt.model_depth,
            n_classes=opt.n_classes,
            n_input_channels=opt.n_input_channels,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool,
            widen_factor=opt.resnet_widen_factor,
        )
    elif opt.model == 'wideresnet':
        model = wide_resnet.generate_model(
            model_depth=opt.model_depth,
            k=opt.wide_resnet_k,
            n_classes=opt.n_classes,
            n_input_channels=opt.n_input_channels,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool
        )
    elif opt.model == 'resnext':
        model = resnext.generate_model(
            model_depth=opt.model_depth,
            cardinality=opt.resnext_cardinality,
            n_classes=opt.n_classes,
            n_input_channels=opt.n_input_channels,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool,
        )
    elif opt.model == 'preresnet':
        model = pre_act_resnet.generate_model(
            model_depth=opt.model_depth,
            n_classes=opt.n_classes,
            n_input_channels=opt.n_input_channels,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool,
        )
    elif opt.model == 'densenet':
        model = densenet.generate_model(
            model_depth=opt.model_depth,
            n_classes=opt.n_classes,
            n_input_channels=opt.n_input_channels,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool,
        )
    else:
        raise ValueError

    if pretrain:
        assert opt.pretrain_name is not None
        pretrain_path = opt.pretrain_root / opt.pretrain_name / 'state.pth'
        # pretrain loading wrapper
        if opt.rank == 0:
            logging.info(f'load pre-trained weights from {pretrain_path}')
        pretrained_state_dict: OrderedDict[str, Tensor] = torch.load(pretrain_path)['state_dict']
        skip_keys = ['fc']
        update_state_dict = OrderedDict({
            key: weight for key, weight in pretrained_state_dict.items()
            if not any(skip_key in key for skip_key in skip_keys)
        })
        missing_keys, unexpected_keys = model.load_state_dict(update_state_dict, strict=False)
        assert len(unexpected_keys) == 0

    return model.to(opt.device)

def setup_finetune(model, model_name, n_finetune_classes):
    tmp_model = model
    if model_name == 'densenet':
        tmp_model.fc = nn.Linear(tmp_model.fc.in_features, n_finetune_classes)
    else:
        tmp_model.fc = nn.Linear(tmp_model.fc.in_features, n_finetune_classes)

def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes: Optional[int] = None):
    if pretrain_path:
        pretrain = torch.load(pretrain_path, map_location='cpu')
        import torch.distributed as dist
        if dist.get_rank() == 0:
            logging.info('loaded pretrained model from {}\n'.format(pretrain_path))
        model.load_state_dict(pretrain['state_dict'])

        if n_finetune_classes:
            setup_finetune(model, model_name, n_finetune_classes)
    return model

def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model