from __future__ import absolute_import
from __future__ import division

__all__ = ['bdnet', 'bdnet_neck', 'bdnet_doublebot_botstream', 'bdnet_neck_doublebot_botstream']

import torch
from torch import nn
from torch.nn import functional as F
from .resnet import resnet50_ls
from torchvision.models.resnet import Bottleneck
import random


"""
# batchdrop blocks
"""
class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
    
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h-rh)
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0
            x = x * mask
        return x

class TopBatchDrop(nn.Module):
    def __init__(self, h_ratio):
        super(TopBatchDrop, self).__init__()
        self.h_ratio = h_ratio
    
    def forward(self, x):
        if self.training:
            b, c, h, w = x.size()
            rh = round(self.h_ratio * h)
            act = (x**2).sum(1)
            act = act.view(b, h*w)
            act = F.normalize(act, p=2, dim=1)
            act = act.view(b, h, w)
            max_act, _ = act.max(2)
            ind = torch.argsort(max_act, 1)
            ind = ind[:, -rh:]
            mask = []
            for i in range(b):
                rmask = torch.ones(h)
                rmask[ind[i]] = 0
                mask.append(rmask.unsqueeze(0))
            mask = torch.cat(mask)
            mask = torch.repeat_interleave(mask, w, 1).view(b, h, w)
            mask = torch.repeat_interleave(mask, c, 0).view(b, c, h, w)
            if x.is_cuda: mask = mask.cuda()
            x = x * mask
        return x

class BatchFeatureErase_Basic(nn.Module):
    def __init__(self, channels, h_ratio=0.33, w_ratio=1.):
        super(BatchFeatureErase_Basic, self).__init__()
        self.drop_batch_bottleneck = Bottleneck(channels, 512)
        self.drop_batch_drop = BatchDrop(h_ratio, w_ratio)

    def forward(self, x):
        x = self.drop_batch_bottleneck(x)
        x = self.drop_batch_drop(x)
        return x

class BatchFeatureErase_Top(nn.Module):
    def __init__(self, channels, h_ratio=0.33, w_ratio=1., double_bottleneck = False, planes=512):
        super(BatchFeatureErase_Top, self).__init__()
        if double_bottleneck:
            self.drop_batch_bottleneck = nn.Sequential(
                Bottleneck(channels, planes),
                Bottleneck(channels, planes)
            )
        else:
            self.drop_batch_bottleneck = Bottleneck(channels, planes)

        self.drop_batch_drop_basic = BatchDrop(h_ratio, w_ratio)
        self.drop_batch_drop_top = TopBatchDrop(h_ratio)

    def forward(self, x, drop_top=False, bottleneck_features = False):
        features = self.drop_batch_bottleneck(x)
        if drop_top:
            x = self.drop_batch_drop_top(features)
        else:
            x = self.drop_batch_drop_basic(features)

        if bottleneck_features:
            return x, features
        else:
            return x

"""
# batch dropblock net
"""
class ResNet50_BD(nn.Module):
    """
    """
    def __init__(self, num_classes=0, loss='softmax', neck = False, drop_height_ratio=0.33, drop_width_ratio=1.0, double_bottleneck=False, db_bottleneck_stream=False, **kwargs):
        super(ResNet50_BD, self).__init__()
        self.loss = loss
        self.db_bottleneck_stream = db_bottleneck_stream
        if neck:
            self.neck_global = nn.BatchNorm1d(512)
            self.neck_global.bias.requires_grad_(False)  # no shift
            self.neck_bd = nn.BatchNorm1d(1024)
            self.neck_bd.bias.requires_grad_(False)  # no shift
            self.neck_bd_bottleneck = nn.BatchNorm1d(2048)
            self.neck_bd_bottleneck.bias.requires_grad_(False)  # no shift
        else:
            self.neck_global = None
            self.neck_bd = None
            self.neck_bd_bottleneck = None

        self.reduction_global = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.reduction_bd = nn.Sequential(
            nn.Linear(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier_global = nn.Linear(512, num_classes)
        self.classifier_bd = nn.Linear(1024, num_classes)
        self.batch_drop = BatchFeatureErase_Top(2048, drop_height_ratio, drop_width_ratio, double_bottleneck)
        if self.db_bottleneck_stream:
            self.classifier_bd_bottleneck = nn.Linear(2048, num_classes)
        else:
            self.classifier_bd_bottleneck = None
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

        #b_bd
        if self.db_bottleneck_stream:
            x_drop, x_bd_bottleneck_t = self.batch_drop(x, drop_top=drop_top, bottleneck_features = True)
            x_bd_bottleneck_t = self.avgpool(x_bd_bottleneck_t).view(x_bd_bottleneck_t.size()[:2])
            if self.neck_bd_bottleneck:
                x_bd_bottleneck_x = self.neck_bd_bottleneck(x_bd_bottleneck_t)
            else:
                x_bd_bottleneck_x = x_bd_bottleneck_t
            x_bd_bottleneck_prelogits = self.classifier_bd_bottleneck(x_bd_bottleneck_x)
        else:
            x_drop = self.batch_drop(x, drop_top=drop_top)
            x_bd_bottleneck_prelogits, x_bd_bottleneck_t = None, None

        #global
        x = self.avgpool(x)
        x_t = self.reduction_global(x)
        x_t = x_t.view(x_t.size()[:2])
        if self.neck_global:
            x_x = self.neck_global(x_t)
        else:
            x_x = x_t
        x_prelogits = self.classifier_global(x_x)

        #db
        x_drop = self.maxpool(x_drop).view(x_drop.size()[:2])
        x_drop_t = self.reduction_bd(x_drop)
        if self.neck_bd:
            x_drop_x = self.neck_bd(x_drop_t)
        else:
            x_drop_x = x_drop_t
        x_drop_prelogits = self.classifier_bd(x_drop_x)

        if not self.training:
            if torch.is_tensor(x_bd_bottleneck_prelogits):
                return torch.cat((x_x, x_drop_x, x_bd_bottleneck_x), dim=1)
            else:
                return torch.cat((x_x, x_drop_x), dim=1)

        if self.loss == 'triplet_xent_batchdrop':
            return x_prelogits, x_t, x_drop_prelogits, x_drop_t
        elif self.loss == 'triplet_xent_top_batchdrop':
            return x_prelogits, x_t, x_drop_prelogits, x_drop_t, x_bd_bottleneck_prelogits, x_bd_bottleneck_t
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def bdnet(num_classes, loss='softmax', pretrained=True, **kwargs):
    """
    Replicates work of Dai et. al. Batch DropBlock Network for Person Re-identification and Beyond
    """
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck = False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        double_bottleneck=False,
        db_bottleneck_stream=False,
        **kwargs
    )
    return model

def bdnet_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck = True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        double_bottleneck=False,
        db_bottleneck_stream=False,
        **kwargs
    )
    return model

def bdnet_doublebot_botstream(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck = False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        double_bottleneck=True,
        db_bottleneck_stream=True,
        **kwargs
    )
    return model

def bdnet_neck_doublebot_botstream(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet50_BD(
        num_classes=num_classes,
        loss=loss,
        neck = True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        double_bottleneck=True,
        db_bottleneck_stream=True,
        **kwargs
    )
    return model

