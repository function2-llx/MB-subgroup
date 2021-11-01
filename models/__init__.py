import logging
from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor, nn

from utils.conf import Conf
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

def generate_model(conf: Conf, pretrain: bool = True) -> Backbone:
    if conf.model == 'unet':
        model = UNet(
            dimensions=3,
            in_channels=3,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            n_classes=conf.n_classes,
        )
    elif conf.model == 'resnet':
        model = resnet.generate_model(
            model_depth=conf.model_depth,
            n_classes=len(conf.subgroups),
            n_input_channels=len(conf.protocols),
            shortcut_type=conf.resnet_shortcut,
            conv1_t_size=conf.conv1_t_size,
            conv1_t_stride=conf.conv1_t_stride,
            no_max_pool=conf.no_max_pool,
            widen_factor=conf.resnet_widen_factor,
            num_seg=len(conf.segs),
            recons=conf.recons,
        )
    elif conf.model == 'resnet2p1d':
        model = resnet2p1d.generate_model(
            model_depth=conf.model_depth,
            n_classes=conf.n_classes,
            n_input_channels=conf.n_input_channels,
            shortcut_type=conf.resnet_shortcut,
            conv1_t_size=conf.conv1_t_size,
            conv1_t_stride=conf.conv1_t_stride,
            no_max_pool=conf.no_max_pool,
            widen_factor=conf.resnet_widen_factor,
        )
    elif conf.model == 'wideresnet':
        model = wide_resnet.generate_model(
            model_depth=conf.model_depth,
            k=conf.wide_resnet_k,
            n_classes=conf.n_classes,
            n_input_channels=conf.n_input_channels,
            shortcut_type=conf.resnet_shortcut,
            conv1_t_size=conf.conv1_t_size,
            conv1_t_stride=conf.conv1_t_stride,
            no_max_pool=conf.no_max_pool
        )
    elif conf.model == 'resnext':
        model = resnext.generate_model(
            model_depth=conf.model_depth,
            cardinality=conf.resnext_cardinality,
            n_classes=conf.n_classes,
            n_input_channels=conf.n_input_channels,
            shortcut_type=conf.resnet_shortcut,
            conv1_t_size=conf.conv1_t_size,
            conv1_t_stride=conf.conv1_t_stride,
            no_max_pool=conf.no_max_pool,
        )
    elif conf.model == 'preresnet':
        model = pre_act_resnet.generate_model(
            model_depth=conf.model_depth,
            n_classes=conf.n_classes,
            n_input_channels=conf.n_input_channels,
            shortcut_type=conf.resnet_shortcut,
            conv1_t_size=conf.conv1_t_size,
            conv1_t_stride=conf.conv1_t_stride,
            no_max_pool=conf.no_max_pool,
        )
    elif conf.model == 'densenet':
        model = densenet.generate_model(
            model_depth=conf.model_depth,
            n_classes=conf.n_classes,
            n_input_channels=conf.n_input_channels,
            conv1_t_size=conf.conv1_t_size,
            conv1_t_stride=conf.conv1_t_stride,
            no_max_pool=conf.no_max_pool,
        )
    else:
        raise ValueError

    if pretrain:
        assert conf.pretrain_name is not None
        pretrain_path = conf.pretrain_root / conf.pretrain_name / 'state.pth'
        # pretrain loading wrapper
        if conf.rank == 0:
            logging.info(f'load pre-trained weights from {pretrain_path}')
        pretrained_state_dict: OrderedDict[str, Tensor] = torch.load(pretrain_path)['state_dict']
        skip_keys = ['fc']
        update_state_dict = OrderedDict({
            key: weight for key, weight in pretrained_state_dict.items()
            if not any(skip_key in key for skip_key in skip_keys)
        })
        missing_keys, unexpected_keys = model.load_state_dict(update_state_dict, strict=False)

    return model.to(conf.device)

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
