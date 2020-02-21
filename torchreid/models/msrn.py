from __future__ import absolute_import
from __future__ import division

__all__ = ['msrn_bd_botdropfeat_doubot', 'msrn_bd_neck_botdropfeat_doubot']

import torch
from torch import nn
from torch.nn import functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=False, activation = 'leakyrelu', dropout = False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope = 0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == None:
            self.activation = None
        else:
            raise ValueError('Not a valid activation, received {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class MSRB(nn.Module):
    def __init__(self, n_feats=64):
        super(MSRB, self).__init__()
        self.conv_1_stream_1 = Conv2d(n_feats, n_feats, 3, stride = 1, bn = True, activation = 'relu')
        self.conv_1_stream_2 = Conv2d(n_feats, n_feats, 5, stride = 1, bn = True, activation = 'relu')
        self.conv_2_stream_1 = Conv2d(2*n_feats, 2*n_feats, 3, stride = 1, bn = True, activation = 'relu')
        self.conv_2_stream_2 = Conv2d(2*n_feats, 2*n_feats, 5, stride = 1, bn = True, activation = 'relu')
        self.confusion = Conv2d(4*n_feats, n_feats, 1, stride = 1, bn = False, activation = None)

    def forward(self, x):
        x_1 = self.conv_1_stream_1(x)
        x_2 = self.conv_1_stream_2(x)
        x_concat = torch.cat([x_1, x_2], dim=1)
        x_1 = self.conv_2_stream_1(x_concat)
        x_2 = self.conv_2_stream_2(x_concat)
        x_concat = torch.cat([x_1, x_2], dim=1)
        output = self.confusion(x_concat)
        output += x
        return output

class MSRN(nn.Module):
    def __init__(self, in_n_feats=64, out_n_feats=2048, n_blocks=8):
        print("Creating MSRN with {} blocks and {} feature channels".format(n_blocks, in_n_feats))
        super(MSRN, self).__init__()
        self.n_feats = in_n_feats
        self.n_blocks = n_blocks

        self.head = Conv2d(3, self.n_feats, 3, stride = 1, bn = True, activation = 'relu')
        self.body = nn.ModuleList([MSRB(n_feats = self.n_feats) for i in range(n_blocks)])
        self.tail = Conv2d(self.n_feats * (n_blocks + 1), out_n_feats, 1, stride = 1, bn = False, activation = None)

    def forward(self, x, **kwargs):
        x = self.head(x)

        MSRB_out = [x]
        for i in range(self.n_blocks):
            x = self.body[i](x)
            MSRB_out.append(x)

        res = torch.cat(MSRB_out,dim=1)
        output = self.tail(res)
        return output

from .fgnet import BatchFeatureErase, BatchFeatureErase_Basic, BatchFeatureErase_Top

class MSRN_BD(nn.Module):
    """
    """
    def __init__(self, num_classes=0, loss='softmax', neck = False, drop_height_ratio=0.33, drop_width_ratio=1.0, double_bottleneck=False, drop_bottleneck_features=False, in_n_feats=64, n_blocks=8, **kwargs):
        super(MSRN_BD, self).__init__()
        self.loss = loss
        self.drop_bottleneck_features = drop_bottleneck_features
        self.base_channel_size = 512
        if neck:
            self.bottleneck_global = nn.BatchNorm1d(512)
            self.bottleneck_global.bias.requires_grad_(False)  # no shift
            self.bottleneck_db = nn.BatchNorm1d(1024)
            self.bottleneck_db.bias.requires_grad_(False)  # no shift
            self.bottleneck_drop_bottleneck_features = nn.BatchNorm1d(self.base_channel_size)
            self.bottleneck_drop_bottleneck_features.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck_global = None
            self.bottleneck_db = None
            self.bottleneck_drop_bottleneck_features = None

        self.reduction_global = nn.Sequential(
            nn.Conv2d(self.base_channel_size, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.reduction_db = nn.Sequential(
            nn.Linear(self.base_channel_size, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier_global = nn.Linear(512, num_classes)
        self.classifier_db = nn.Linear(1024, num_classes)
        self.batch_drop = BatchFeatureErase_Top(self.base_channel_size, drop_height_ratio, drop_width_ratio, double_bottleneck, planes=self.base_channel_size//4)
        if self.drop_bottleneck_features:
            self.classifier_drop_bottleneck = nn.Linear(self.base_channel_size, num_classes)
        else:
            self.classifier_drop_bottleneck = None
        self.base = MSRN(in_n_feats=in_n_feats, out_n_feats=self.base_channel_size, n_blocks=n_blocks)
        self._init_params()

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

        if self.drop_bottleneck_features:
            drop_x, t_drop_bottleneck_features = self.batch_drop(x, drop_top=drop_top, bottleneck_features = True)
            t_drop_bottleneck_features = self.avgpool(t_drop_bottleneck_features).view(t_drop_bottleneck_features.size()[:2])
            if self.bottleneck_drop_bottleneck_features:
                x_drop_bottleneck_features = self.bottleneck_drop_bottleneck_features(t_drop_bottleneck_features)
            else:
                x_drop_bottleneck_features = t_drop_bottleneck_features
            x_drop_bottleneck_features = self.classifier_drop_bottleneck(x_drop_bottleneck_features)
        else:
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
        if self.loss == 'triplet_dropbatch_dropbotfeatures':
            return x_prelogits, t_x, (x_drop_prelogits, t_drop_x), x_drop_bottleneck_features, t_drop_bottleneck_features
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def msrn_bd_botdropfeat_doubot(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = MSRN_BD(
        num_classes=num_classes,
        loss=loss,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        in_n_feats=32,
        n_blocks=4,
        **kwargs
    )
    return model

def msrn_bd_neck_botdropfeat_doubot(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = MSRN_BD(
        num_classes=num_classes,
        loss=loss,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        drop_bottleneck_features=True,
        double_bottleneck=True,
        in_n_feats=32,
        n_blocks=4,
        **kwargs
    )
    return model