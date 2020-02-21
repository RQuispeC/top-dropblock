"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import absolute_import
from __future__ import division

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d',
           'resnext101_32x8d', 'resnet50_fc512', 'resnet50m', 'resnet50m_neck', 'resnet50m_ls_neck', 'resnet50_ls', 'resnet50m_bd_ls_neck', 'resnet50_bd', 'resnet50_bd_neck', 'resnet50_bd_botdropfeat', 'resnet50_bd_neck_botdropfeat', 'resnet50_bd_botdropfeat_doubot', 'resnet50_bd_neck_botdropfeat_doubot', 'resnet50_bd_botdropfeat_inter', 'resnet50_bd_neck_botdropfeat_inter', 'resnet50_bd_botdropfeat_doubot_inter', 'resnet50_bd_neck_botdropfeat_doubot_inter', 'resnet50_bd_neck_botdropfeat_doubot_pose', 'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_1', 'resnet50_bd_neck_botdropfeat_doubot_pose_13_conv_1', 'resnet50_bd_neck_botdropfeat_doubot_pose_13_conv_2', 'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_4', 'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_5', 'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_6', 'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_7', 'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_8', 'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_9', 'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_10', 'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_11']

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Residual network.
    
    Reference:
        - He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
        - Xie et al. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.

    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnext50_32x4d``: ResNeXt50.
        - ``resnext101_32x8d``: ResNeXt101.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(self, num_classes, loss, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, last_stride=2, fc_dims=None, dropout_p=None, neck=None, **kwargs):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride,
                                       dilate=replace_stride_with_dilation[2])
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        if neck:
            self.bottleneck = nn.BatchNorm1d(self.feature_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck = None

        self._init_params()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        
        self.feature_dim = fc_dims[-1]
        
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x, return_featuremaps=False):
        f = self.featuremaps(x)
        if return_featuremaps:
            return f
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        
        if self.fc is not None:
            v = self.fc(v)
        
        if self.bottleneck:
            t = self.bottleneck(v)
        else:
            t = v

        if not self.training:
            return t
        
        y = self.classifier(t)
        
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


"""ResNet"""
def resnet18(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


def resnet34(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])
    return model


def resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def resnet50_ls(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def resnet101(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])
    return model


def resnet152(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])
    return model


"""ResNeXt"""
def resnext50_32x4d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        groups=32,
        width_per_group=4,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnext50_32x4d'])
    return model


def resnext101_32x8d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        groups=32,
        width_per_group=8,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnext101_32x8d'])
    return model


"""
ResNet + FC
"""
def resnet50_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

class ResNet50M(nn.Module):
    """ResNet50 + mid-level features.
    Reference:
    Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
    Cross-Domain Instance Matching. arXiv:1711.08106.
    """
    def __init__(self, num_classes=0, loss='softmax', neck = False, last_stride=False, **kwargs):
        super(ResNet50M, self).__init__()
        self.loss = loss
        if neck:
            self.bottleneck = nn.BatchNorm1d(3072)
            self.bottleneck.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck = None
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3072, num_classes)
        self._init_params()


        if last_stride:
            resnet = resnet50_ls(num_classes, pretrained=True) #resnet50 with last stride = 1
        else:
            resnet = resnet50(num_classes, pretrained=True)
        base = nn.Sequential(*list(resnet.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_featuremaps = False):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        if return_featuremaps:
            return x5c

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))
        
        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        feature_t = torch.cat((x5c_feat, midfeat), dim=1)
        if self.bottleneck:
            combofeat = self.bottleneck(feature_t)
        else:
            combofeat = feature_t

        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)

        if self.loss == 'softmax':
            return prelogits
        elif self.loss == 'triplet':
            return prelogits, feature_t
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

from .fgnet import BatchFeatureErase, BatchFeatureErase_Basic, BatchFeatureErase_Top
class ResNet50M_BD(nn.Module):
    """ResNet50 + mid-level features.
    Reference:
    Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
    Cross-Domain Instance Matching. arXiv:1711.08106.
    """
    def __init__(self, num_classes=0, loss='softmax', neck = False, last_stride=False, drop_height_ratio=0.33, drop_width_ratio=1.0, **kwargs):
        super(ResNet50M_BD, self).__init__()
        self.loss = loss
        if neck:
            self.bottleneck = nn.BatchNorm1d(3072)
            self.bottleneck.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck = None
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3072, num_classes)
        self.batch_drop = BatchFeatureErase(num_classes, 2048, drop_height_ratio, drop_width_ratio)
        self._init_params()


        if last_stride:
            resnet = resnet50_ls(num_classes, pretrained=True) #resnet50 with last stride = 1
        else:
            resnet = resnet50(num_classes, pretrained=True)
        base = nn.Sequential(*list(resnet.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_featuremaps = False):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        if return_featuremaps:
            return x5c
        drop_features = self.batch_drop(x5c)
        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))
        
        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        feature_t = torch.cat((x5c_feat, midfeat), dim=1)
        if self.bottleneck:
            combofeat = self.bottleneck(feature_t)
        else:
            combofeat = feature_t

        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)

        if self.loss == 'softmax':
            return prelogits
        elif self.loss == 'triplet':
            return prelogits, feature_t
        elif self.loss == 'triplet_dropbatch':
            return prelogits, feature_t, drop_features
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50_BD(nn.Module):
    """
    """
    def __init__(self, num_classes=0, loss='softmax', neck = False, drop_height_ratio=0.33, drop_width_ratio=1.0, **kwargs):
        super(ResNet50_BD, self).__init__()
        self.loss = loss
        if neck:
            self.bottleneck_global = nn.BatchNorm1d(512)
            self.bottleneck_global.bias.requires_grad_(False)  # no shift
            self.bottleneck_db = nn.BatchNorm1d(1024)
            self.bottleneck_db.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck_global = None
            self.bottleneck_db = None

        self.reduction_global = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.reduction_db = nn.Sequential(
            nn.Linear(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier_global = nn.Linear(512, num_classes)
        self.classifier_db = nn.Linear(1024, num_classes)
        self.batch_drop = BatchFeatureErase_Top(2048, drop_height_ratio, drop_width_ratio)
        self._init_params()

        resnet = resnet50_ls(num_classes, pretrained=True) #resnet50 with last stride = 1
        self.base = nn.Sequential(*list(resnet.children())[:-2])

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_featuremaps = False, drop_top=False):
        x = self.base(x)
        if return_featuremaps:
            return x
        drop_x = self.batch_drop(x, drop_top=drop_top)

        #global
        x = self.avgpool(x)
        t_x = self.reduction_global(x)
        t_x = t_x.view(t_x.size()[:2])
        if self.bottleneck_global:
            x_x = self.bottleneck_global(t_x)
        else:
            x_x = t_x
        x_prelogits = self.classifier_global(x_x)

        #db
        drop_x = self.maxpool(drop_x).view(drop_x.size()[:2])
        t_drop_x = self.reduction_db(drop_x)
        if self.bottleneck_db:
            x_drop_x = self.bottleneck_db(t_drop_x)
        else:
            x_drop_x = t_drop_x
        x_drop_prelogits = self.classifier_db(x_drop_x)

        if not self.training:
            return torch.cat((x_x, x_drop_x), dim=1)

        if self.loss == 'triplet_dropbatch':
            return x_prelogits, t_x, (x_drop_prelogits, t_drop_x)
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class intermediate_ResNet50_BD(nn.Module):
    def __init__(self, num_classes):
        super(intermediate_ResNet50_BD, self).__init__()
        resnet = resnet50_ls(num_classes, pretrained=True)
        base = nn.Sequential(*list(resnet.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5 = base[7]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.combine = nn.Sequential(
            nn.Conv2d(2048+1024+512, 2048, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)#[b, 512, 2y, 2x]
        x4 = self.layers4(x3)#[b, 1024, y, x]
        x5 = self.layers5(x4)#[b, 2048, y, x]
        #x3 = self.maxpool(x3)
        x4 = F.interpolate(x4, x3.size()[2:], mode='bilinear')
        x5 = F.interpolate(x5, x3.size()[2:], mode='bilinear')
        x = torch.cat((x3, x4, x5), dim=1)
        x = self.combine(x)
        return x

class ResNet50_BD(nn.Module):
    """
    """
    def __init__(self, num_classes=0, loss='softmax', neck = False, drop_height_ratio=0.33, drop_width_ratio=1.0, double_bottleneck=False, drop_bottleneck_features=False, intermediate=False, pose_invariant=[], batch_num_classes=-1, pose_invariant_mode='', **kwargs):
        super(ResNet50_BD, self).__init__()
        self.pose_invariant = pose_invariant
        self.pose_invariant_mode = pose_invariant_mode

        feat_size = []
        pose_input_size = []
        if pose_invariant_mode.startswith("fc"):
            if "global" in self.pose_invariant: feat_size.append((512//3)*2)
            else: feat_size.append(512)
            if "db" in self.pose_invariant: feat_size.append((1024//3)*2)
            else: feat_size.append(1024)
            if "drop_bottleneck_features" in self.pose_invariant: feat_size.append((2048//3)*2)
            else: feat_size.append(2048)
            pose_input_size.append(512)
            pose_input_size.append(1024)
            pose_input_size.append(2048)
        elif pose_invariant_mode.startswith("conv"):
            if "global" in self.pose_invariant: feat_size.append((2048//3)*2)
            else: feat_size.append(512)
            if "db" in self.pose_invariant: feat_size.append((2048//3)*2)
            else: feat_size.append(1024)
            if "drop_bottleneck_features" in self.pose_invariant: feat_size.append((2048//3)*2)
            else: feat_size.append(2048)
            pose_input_size.append(2048)
            pose_input_size.append(2048)
            pose_input_size.append(2048)
        else:
            feat_size.append(512)
            feat_size.append(1024)
            feat_size.append(2048)

        self.loss = loss
        self.drop_bottleneck_features = drop_bottleneck_features
        if neck:
            self.bottleneck_global = nn.BatchNorm1d(feat_size[0])
            self.bottleneck_global.bias.requires_grad_(False)  # no shift
            self.bottleneck_db = nn.BatchNorm1d(feat_size[1])
            self.bottleneck_db.bias.requires_grad_(False)  # no shift
            self.bottleneck_drop_bottleneck_features = nn.BatchNorm1d(feat_size[2])
            self.bottleneck_drop_bottleneck_features.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck_global = None
            self.bottleneck_db = None
            self.bottleneck_drop_bottleneck_features = None

        self.reduction_global = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.reduction_db = nn.Sequential(
            nn.Linear(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier_global = nn.Linear(feat_size[0], num_classes)
        self.classifier_db = nn.Linear(feat_size[1], num_classes)
        self.batch_drop = BatchFeatureErase_Top(2048, drop_height_ratio, drop_width_ratio, double_bottleneck)
        if self.drop_bottleneck_features:
            self.classifier_drop_bottleneck = nn.Linear(feat_size[2], num_classes)
        else:
            self.classifier_drop_bottleneck = None
        if self.pose_invariant:
            if batch_num_classes == -1: raise KeyError("Invalid value for 'batch_num_classes': {}".format(batch_num_classes))
            if "global" in self.pose_invariant: self.pose_invariant_global = PoseInvariantNet(batch_num_classes, pose_input_size[0], mode=pose_invariant_mode, pooling='avg')
            else: self.pose_invariant_global = None
            if "db" in self.pose_invariant: self.pose_invariant_db = PoseInvariantNet(batch_num_classes, pose_input_size[1], mode=pose_invariant_mode, pooling='max')
            else: self.pose_invariant_db = None
            if self.drop_bottleneck_features and "drop_bottleneck_features" in self.pose_invariant:
                self.pose_invariant_drop_bottleneck = PoseInvariantNet(batch_num_classes, pose_input_size[2], mode=pose_invariant_mode, pooling='avg')
            else:
                self.pose_invariant_drop_bottleneck = None
        else:
            self.pose_invariant_global = None
            self.pose_invariant_db = None
            self.pose_invariant_drop_bottleneck = None
        self._init_params()

        self.intermediate = intermediate
        if intermediate:
            self.base = intermediate_ResNet50_BD(num_classes)
        else:
            resnet = resnet50_ls(num_classes, pretrained=True) #resnet50 with last stride = 1
            self.base = nn.Sequential(*list(resnet.children())[:-2])

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_featuremaps = False, drop_top=False):
        x = self.base(x)
        if return_featuremaps:
            return x

        #b_db
        if self.drop_bottleneck_features:
            drop_x, t_drop_bottleneck_features = self.batch_drop(x, drop_top=drop_top, bottleneck_features = True)
            if self.pose_invariant_drop_bottleneck:
                if self.pose_invariant_mode.startswith("fc"):
                    t_drop_bottleneck_features = self.avgpool(t_drop_bottleneck_features).view(t_drop_bottleneck_features.size()[:2])
                    x_dbf, x_prime_dbf, h1_dbf, h2_dbf, h3_dbf, h4_dbf, t_drop_bottleneck_features = self.pose_invariant_drop_bottleneck(t_drop_bottleneck_features)
                elif self.pose_invariant_mode.startswith("conv"):
                    x_dbf, x_prime_dbf, h1_dbf, h2_dbf, h3_dbf, h4_dbf, t_drop_bottleneck_features = self.pose_invariant_drop_bottleneck(t_drop_bottleneck_features)
            else:
                t_drop_bottleneck_features = self.avgpool(t_drop_bottleneck_features).view(t_drop_bottleneck_features.size()[:2])
                x_dbf, x_prime_dbf, h1_dbf, h2_dbf, h3_dbf, h4_dbf = None, None, None, None, None, None
            if self.bottleneck_drop_bottleneck_features:
                x_drop_bottleneck_features = self.bottleneck_drop_bottleneck_features(t_drop_bottleneck_features)
            else:
                x_drop_bottleneck_features = t_drop_bottleneck_features
            x_drop_bottleneck_features = self.classifier_drop_bottleneck(x_drop_bottleneck_features)
        else:
            drop_x = self.batch_drop(x, drop_top=drop_top)

        #global
        if self.pose_invariant_global:
            if self.pose_invariant_mode.startswith("fc"):
                x = self.avgpool(x)
                t_x = self.reduction_global(x)
                t_x = t_x.view(t_x.size()[:2])
                x_g, x_prime_g, h1_g, h2_g, h3_g, h4_g, t_x = self.pose_invariant_global(t_x)
            elif self.pose_invariant_mode.startswith("conv"):
                x_g, x_prime_g, h1_g, h2_g, h3_g, h4_g, t_x = self.pose_invariant_global(x)
        else:
            x = self.avgpool(x)
            t_x = self.reduction_global(x)
            t_x = t_x.view(t_x.size()[:2])
            x_g, x_prime_g, h1_g, h2_g, h3_g, h4_g = None, None, None, None, None, None
        if self.bottleneck_global:
            x_x = self.bottleneck_global(t_x)
        else:
            x_x = t_x
        x_prelogits = self.classifier_global(x_x)

        #db
        if self.pose_invariant_db:
            if self.pose_invariant_mode.startswith("fc"):
                drop_x = self.maxpool(drop_x).view(drop_x.size()[:2])
                t_drop_x = self.reduction_db(drop_x)
                x_d, x_prime_d, h1_d, h2_d, h3_d, h4_d, t_drop_x = self.pose_invariant_db(t_drop_x)
            elif self.pose_invariant_mode.startswith("conv"):
                x_d, x_prime_d, h1_d, h2_d, h3_d, h4_d, t_drop_x = self.pose_invariant_db(drop_x)
        else:
            drop_x = self.maxpool(drop_x).view(drop_x.size()[:2])
            t_drop_x = self.reduction_db(drop_x)
            x_d, x_prime_d, h1_d, h2_d, h3_d, h4_d = None, None, None, None, None, None
        if self.bottleneck_db:
            x_drop_x = self.bottleneck_db(t_drop_x)
        else:
            x_drop_x = t_drop_x
        x_drop_prelogits = self.classifier_db(x_drop_x)

        if not self.training:
            #return torch.cat((x_x, x_drop_x), dim=1)
            #return torch.cat((x_x, x_drop_bottleneck_features), dim=1)
            return torch.cat((x_x, x_drop_x, x_drop_bottleneck_features), dim=1)

        if self.loss == 'triplet_dropbatch':
            return x_prelogits, t_x, (x_drop_prelogits, t_drop_x)
        elif self.loss == 'triplet_dropbatch_dropbotfeatures' or self.loss == 'reference_dropbatch_dropbotfeatures' or self.loss == 'group_triplet_dropbatch_dropbotfeatures' or self.loss == 'focal_triplet_dropbatch_dropbotfeatures' or self.loss == 'dependency_triplet_dropbatch_dropbotfeatures' or self.loss == 'cluster_triplet_dropbatch_dropbotfeatures' or self.loss ==  'cluster_dependency_triplet_dropbatch_dropbotfeatures':
            return x_prelogits, t_x, (x_drop_prelogits, t_drop_x), x_drop_bottleneck_features, t_drop_bottleneck_features
        elif self.loss == 'pose_triplet_dropbatch_dropbotfeatures':
            return x_prelogits, t_x, x_drop_prelogits, t_drop_x, x_drop_bottleneck_features, t_drop_bottleneck_features, [x_g, x_prime_g, h1_g, h2_g, h3_g, h4_g], [x_d, x_prime_d, h1_d, h2_d, h3_d, h4_d], [x_dbf, x_prime_dbf, h1_dbf, h2_dbf, h3_dbf, h4_dbf]
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class PoseInvariantNet(nn.Module):
    """
    creates pose invariant streams based on:
    Reconstruction-Based Disentanglement for Pose-invariant Face Recognition, Peng et.al.
    Joint Discriminative and Generative Learning for Person Re-identification Zheng et. al.
    """
    def __init__(self, batch_num_classes=0, feature_dim=512, mode = 'fc_1', pooling = None):
        super(PoseInvariantNet, self).__init__()
        self.mode = mode
        self.batch_num_classes = batch_num_classes
        self.feature_dim = feature_dim
        self.id_feature_dim = (feature_dim // 3) * 2
        self.not_id_feature_dim = feature_dim - self.id_feature_dim
        if pooling == None: self.pooling = None
        elif pooling == 'avg': self.pooling = nn.AdaptiveAvgPool2d((1,1))
        elif pooling == 'max': self.pooling = nn.AdaptiveMaxPool2d((1,1))
        else: raise KeyError("Unsupported mode for pooling: {}".format(pooling))
        if mode == 'fc_1':
            self.id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.id_feature_dim)
            )
            self.not_id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.not_id_feature_dim)
            )
            self.join_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim)
            )
        elif mode == 'fc_2':
            self.id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.id_feature_dim),
                nn.Linear(self.id_feature_dim, self.id_feature_dim)
            )
            self.not_id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.not_id_feature_dim),
                nn.Linear(self.not_id_feature_dim, self.not_id_feature_dim)
            )
            self.join_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.Linear(self.feature_dim, self.feature_dim)
            )
        elif mode == 'fc_3':
            self.id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.id_feature_dim),
                nn.Linear(self.id_feature_dim, self.id_feature_dim),
                nn.Linear(self.id_feature_dim, self.id_feature_dim)
            )
            self.not_id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.not_id_feature_dim),
                nn.Linear(self.not_id_feature_dim, self.not_id_feature_dim),
                nn.Linear(self.not_id_feature_dim, self.not_id_feature_dim)
            )
            self.join_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.Linear(self.feature_dim, self.feature_dim)
            )
        elif mode == 'fc_4':
            self.id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.id_feature_dim)
            )
            self.not_id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.not_id_feature_dim)
            )
            self.join_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.Linear(self.feature_dim, self.feature_dim)
            )
        elif mode == 'fc_5':
            self.id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.id_feature_dim)
            )
            self.not_id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.not_id_feature_dim)
            )
            self.join_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.Linear(self.feature_dim, self.feature_dim)
            )
        elif mode == 'fc_6':
            self.id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.id_feature_dim),
                nn.Linear(self.id_feature_dim, self.id_feature_dim)
            )
            self.not_id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.not_id_feature_dim),
                nn.Linear(self.not_id_feature_dim, self.not_id_feature_dim)
            )
            self.join_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim)
            )
        elif mode == 'fc_7':
            self.id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.id_feature_dim),
                nn.Linear(self.id_feature_dim, self.id_feature_dim),
                nn.Linear(self.id_feature_dim, self.id_feature_dim)
            )
            self.not_id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.not_id_feature_dim),
                nn.Linear(self.not_id_feature_dim, self.not_id_feature_dim),
                nn.Linear(self.not_id_feature_dim, self.not_id_feature_dim)
            )
            self.join_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim)
            )
        elif mode == 'fc_8' or mode == 'fc_9' or mode == 'fc_10' or mode == 'fc_11':
            self.id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.id_feature_dim)
            )
            self.not_id_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.not_id_feature_dim)
            )
            self.join_net = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim)
            )
            self.id_to_join_net = nn.Sequential(
                nn.Linear(self.id_feature_dim, self.feature_dim)
            )
            self.not_id_to_join_net = nn.Sequential(
                nn.Linear(self.not_id_feature_dim, self.feature_dim)
            )
            if mode == "fc_11": #fix bias to 1
                self.id_to_join_net[0].bias = nn.Parameter(torch.ones_like(self.id_to_join_net[0].bias))
                self.id_to_join_net[0].bias.requires_grad_(False)
                self.not_id_to_join_net[0].bias = nn.Parameter(torch.ones_like(self.not_id_to_join_net[0].bias))
                self.not_id_to_join_net[0].bias.requires_grad_(False)

        elif mode == 'conv_1':
            self.id_net = conv1x1(self.feature_dim, self.id_feature_dim)
            self.not_id_net = conv1x1(self.feature_dim, self.not_id_feature_dim)
            self.join_net = conv1x1(self.feature_dim, self.feature_dim)
        elif mode == 'conv_2':
            self.id_net = conv3x3(self.feature_dim, self.id_feature_dim)
            self.not_id_net = conv3x3(self.feature_dim, self.not_id_feature_dim)
            self.join_net = conv3x3(self.feature_dim, self.feature_dim)
        else:
            raise KeyError("Unsupported mode for pose: {}".format(mode))
        self.cat_join = ['fc_1', 'fc_2', 'fc_3', 'fc_4', 'fc_5', 'fc_6', 'fc_7', 'conv_1', 'conv_2']
        self.prod_join = ['fc_8', 'fc_9', 'fc_10', 'fc_11']
    def forward(self, x):
        """
        x a torch tensor with shape (batch_size, feature_dim)
        we asume that it has self.batch_num_classes different ids and items with same id are llocated adjacently
        """
        if not self.training:
            x = self.id_net(x)
            if self.mode.startswith("conv"):
                x = self.pooling(x)
                x = x.view(x.size()[:2])
            return None, None, None, None, None, None, x

        if self.mode.startswith("fc"):
            b_dim, f_dim =  x.size()
            elems_per_class = b_dim//self.batch_num_classes
            if elems_per_class > 0:
                x_prime = torch.roll(x.view(self.batch_num_classes, elems_per_class, f_dim), shifts=1, dims=1).view(b_dim, f_dim) #for each item inside the batch, get annother instance with same id, thus ID(x[i]) == ID(x_prime[i]) and x[i] != x_prime[i]
            else:
                x_prime = x
        elif self.mode.startswith("conv"):
            b, c, h, w = x.size()
            elems_per_class = b//self.batch_num_classes
            if elems_per_class > 0:
                x_prime = torch.roll(x.view(self.batch_num_classes, elems_per_class, c*h*w), shifts=1, dims=1).view(b, c, h, w) #for each item inside the batch, get annother instance with same id, thus ID(x[i]) == ID(x_prime[i]) and x[i] != x_prime[i]
            else:
                x_prime = x
        x_id =  self.id_net(x)
        x_not_id =  self.not_id_net(x)
        x_prime_id = self.id_net(x_prime)
        x_prime_not_id = self.not_id_net(x_prime)
        if self.mode in self.cat_join:
            h1 = self.join_net(torch.cat((x_id, x_not_id), dim=1))
            h2 = self.join_net(torch.cat((x_id, x_prime_not_id), dim=1))
            h3 = self.join_net(torch.cat((x_prime_id, x_not_id), dim=1))
            h4 = self.join_net(torch.cat((x_prime_id, x_prime_not_id), dim=1))
        elif self.mode in self.prod_join:
            h1_id = self.id_to_join_net(x_id)
            h1_not_id = self.not_id_to_join_net(x_not_id)
            h2_id = self.id_to_join_net(x_id)
            h2_not_id = self.not_id_to_join_net(x_prime_not_id)
            h3_id = self.id_to_join_net(x_prime_id)
            h3_not_id = self.not_id_to_join_net(x_not_id)
            h4_id = self.id_to_join_net(x_prime_id)
            h4_not_id = self.not_id_to_join_net(x_prime_not_id)
            if self.mode == 'fc_8' or self.mode == 'fc_11':
                h1 = h1_id * h1_not_id
                h2 = h2_id * h2_not_id
                h3 = h3_id * h3_not_id
                h4 = h4_id * h4_not_id
            elif self.mode == 'fc_9':
                h1 = self.join_net(h1_id * h1_not_id)
                h2 = self.join_net(h2_id * h2_not_id)
                h3 = self.join_net(h3_id * h3_not_id)
                h4 = self.join_net(h4_id * h4_not_id)
            elif self.mode == 'fc_10':
                h1 = self.join_net(h1_id * h1_not_id) + h1_id * h1_not_id
                h2 = self.join_net(h2_id * h2_not_id) + h2_id * h2_not_id
                h3 = self.join_net(h3_id * h3_not_id) + h3_id * h3_not_id
                h4 = self.join_net(h4_id * h4_not_id) + h4_id * h4_not_id
        if self.mode.startswith("conv"):
            x_id = self.pooling(x_id)
            x_id = x_id.view(x_id.size()[:2])
        return x, x_prime, h1, h2, h3, h4, x_id

def resnet50_bd_botdropfeat_inter(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=False,
        intermediate=True,
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_inter(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=False,
        intermediate=True,
        **kwargs
    )
    return model

def resnet50_bd_botdropfeat_doubot_inter(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        intermediate=True,
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot_inter(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        intermediate=True,
        **kwargs
    )
    return model

def resnet50_bd_botdropfeat(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=False,
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=False,
        **kwargs
    )
    return model

def resnet50_bd_botdropfeat_doubot(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot_pose(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        pose_invariant=["global", "db", "drop_bottleneck_features"],
        pose_invariant_mode='fc_1',
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_1(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        pose_invariant=["global", "drop_bottleneck_features"],
        pose_invariant_mode='fc_1',
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        pose_invariant=["global", "drop_bottleneck_features"],
        pose_invariant_mode='fc_4',
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_5(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        pose_invariant=["global", "drop_bottleneck_features"],
        pose_invariant_mode='fc_5',
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_6(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        pose_invariant=["global", "drop_bottleneck_features"],
        pose_invariant_mode='fc_6',
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_7(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        pose_invariant=["global", "drop_bottleneck_features"],
        pose_invariant_mode='fc_7',
        **kwargs
    )
    return model


def resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_8(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        pose_invariant=["global", "drop_bottleneck_features"],
        pose_invariant_mode='fc_8',
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_9(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        pose_invariant=["global", "drop_bottleneck_features"],
        pose_invariant_mode='fc_9',
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_10(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        pose_invariant=["global", "drop_bottleneck_features"],
        pose_invariant_mode='fc_10',
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_11(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        pose_invariant=["global", "drop_bottleneck_features"],
        pose_invariant_mode='fc_11',
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot_pose_13_conv_1(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        pose_invariant=["global", "drop_bottleneck_features"],
        pose_invariant_mode='conv_1',
        **kwargs
    )
    return model

def resnet50_bd_neck_botdropfeat_doubot_pose_13_conv_2(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        pose_invariant=["global", "drop_bottleneck_features"],
        pose_invariant_mode='conv_2',
        **kwargs
    )
    return model

def resnet50_bd_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model

def resnet50_bd(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model

def resnet50m(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50M(
        num_classes=num_classes,
        loss=loss,
        neck=False,
        **kwargs
    )
    return model

def resnet50m_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50M(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        **kwargs
    )
    return model

def resnet50m_ls_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50M(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        last_stride=True, 
        **kwargs
    )
    return model

def resnet50m_bd_ls_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50M_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        last_stride=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model