from functools import partial
from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks import blocks as monai_blocks
from monai.networks.nets import ResNet

from .backbone import Backbone
from .segresnet import SegResNetOutput
from .utils import permute_img

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(Backbone):
    class Decoder(nn.Module):
        def __init__(self, bottom: nn.Module, up4: nn.Module, up3: nn.Module, up2: nn.Module, up1: nn.Module,
                     out: nn.Module):
            super().__init__()
            self.bottom = bottom
            self.up4 = up4
            self.up3 = up3
            self.up2 = up2
            self.up1 = up1
            self.out = out

        def forward(self, c1, c2, c3, c4, c5):
            up = self.bottom(c5)
            up = self.up4(c5 + up)
            up = self.up3(c4 + up)
            up = self.up2(c3 + up)
            up = self.up1(c2 + up)
            out = self.out(c1 + up)
            return out

    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=3,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type='B',
        widen_factor=1.0,
        n_classes=None,
        num_pretrain_seg=None,
        num_seg=None,
        recons=False,
    ):
        super().__init__()
        # assert n_input_channels == 3
        assert conv1_t_size == 7
        assert conv1_t_stride == 1

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.n_classes = n_classes
        if n_classes:
            self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.num_seg = num_seg
        if num_seg:
            assert block == BasicBlock
            num_cal_seg = num_seg if num_pretrain_seg is None else num_pretrain_seg
            self.seg = ResNet.Decoder(
                self._get_bottom_layer(block_inplanes[3]),
                self._get_up_layer(block_inplanes[3], block_inplanes[2], 2, False),
                self._get_up_layer(block_inplanes[2], block_inplanes[1], 2, False),
                self._get_up_layer(block_inplanes[1], block_inplanes[0], 2, False),
                self._get_up_layer(block_inplanes[0], block_inplanes[0], 1, False),
                self._get_up_layer(block_inplanes[0], num_cal_seg, 2, True),
            )
        else:
            self.seg = None

        if recons:
            self.recons = ResNet.Decoder(
                self._get_bottom_layer(block_inplanes[3]),
                self._get_up_layer(block_inplanes[3], block_inplanes[2], 2, False),
                self._get_up_layer(block_inplanes[2], block_inplanes[1], 2, False),
                self._get_up_layer(block_inplanes[1], block_inplanes[0], 2, False),
                self._get_up_layer(block_inplanes[0], block_inplanes[0], 1, False),
                self._get_up_layer(block_inplanes[0], n_input_channels, 2, True),
            )
        else:
            self.recons = None

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: Union[Sequence[int], int], is_top: bool) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """

        if is_top:
            return nn.Sequential(
                # maxpool
                monai_blocks.Convolution(
                    3,
                    in_channels,
                    in_channels,
                    strides=2,
                    kernel_size=3,
                    conv_only=False,
                    is_transposed=True,
                ),
                # conv1
                monai_blocks.Convolution(
                    3,
                    in_channels,
                    out_channels,
                    strides=(1, 2, 2),
                    # doubt
                    kernel_size=(7, 7, 7),
                    conv_only=True,
                    is_transposed=True,
                )
            )
        else:
            conv: Union[monai_blocks.Convolution, nn.Sequential]
            conv = monai_blocks.Convolution(
                3,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=3,
                conv_only=False,
                is_transposed=True,
            )

            ru = monai_blocks.ResidualUnit(
                3,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=3,
                subunits=1,
                last_conv_only=False,
            )

            return nn.Sequential(conv, ru)

    def _get_bottom_layer(self, in_channels: int, out_channels: int = None) -> nn.Module:
        if out_channels is None:
            out_channels = in_channels
        return monai_blocks.ResidualUnit(
            3,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
        )

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, permute: bool) -> SegResNetOutput:
        if permute:
            x = permute_img(x)
        outputs = {}
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        if not self.no_max_pool:
            c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        outputs['c1'] = c1
        outputs['c2'] = c2
        outputs['c3'] = c3
        outputs['c4'] = c4
        outputs['c5'] = c5

        ret = SegResNetOutput()

        if self.n_classes:
            linear = self.avgpool(c5)
            linear = linear.view(linear.size(0), -1)
            ret.cls_logit = self.fc(linear)

        if self.recons is not None:
            recons = self.recons(c1, c2, c3, c4, c5)
            if permute:
                recons = permute_img(recons, inv=True)
            # outputs['recons'] = permute_img(recons, inv=True)
            outputs['recons'] = recons

        if self.seg is not None:
            if permute:
                ret.seg_logit = permute_img(self.seg(c1, c2, c3, c4, c5)[:, :self.num_seg], inv=True)
            else:
                ret.seg_logit = self.seg(c1, c2, c3, c4, c5)[:, :self.num_seg]

        return ret

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    else:
        raise ValueError

    return model
