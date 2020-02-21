from __future__ import absolute_import
from __future__ import division

__all__ = ['fgnet_1_2', 'fgnet_1_3', 'fgnet_1_4', 'fgnet_2_2', 'fgnet_2_3', 'fgnet_2_4', 'fgnet_2_5', 'fgnet_2_2_resnet50m', 'fgnet_2_3_resnet50m', 'fgnet_2_4_resnet50m', 'fgnet_2_2_resnet50m_neck', 'fgnet_2_3_resnet50m_neck', 'fgnet_2_4_resnet50m_neck', 'fgnet_2_2_ls_neck', 'fgnet_2_3_ls_neck', 'fgnet_2_2_resnet50m_ls_neck', 'fgnet_2_3_resnet50m_ls_neck', 'fgnet_2_2_resnet50m_ls_neck_fc_parts', 'fgnet_2_3_resnet50m_ls_neck_fc_parts', 'fgnet_2_4_resnet50m_ls_neck_fc_parts', 'fgnet_1_1_resnet50m_ls_neck_fc_parts', 'fgnet_1_2_resnet50m_ls_neck_fc_parts', 'fgnet_1_3_resnet50m_ls_neck_fc_parts', 'fgnet_1_1_resnet50m_ls_neck_fc_parts_bd', 'fgnet_1_2_resnet50m_ls_neck_fc_parts_bd', 'fgnet_3_2_resnet50m_ls_neck_fc_parts',
'fgnet_3_3_resnet50m_ls_neck_fc_parts', 'fgnet_1_1_resnet50_fc_parts_bd', 'fgnet_1_2_resnet50_fc_parts_bd', 'fgnet_1_1_resnet50_neck_fc_parts_bd', 'fgnet_1_2_resnet50_neck_fc_parts_bd', 'fgnet_1_1_resnet50_fc_parts_bd_tri', 'fgnet_1_1_resnet50_neck_fc_parts_bd_tri']

import torch
from torch import nn
from torch.nn import functional as F
from .resnet import resnet50_ls, resnet50
from torchvision.models.resnet import Bottleneck
import random

class part_weighter(nn.Module):
    def __init__(self, input_layer_dim, squeeze_scale):
        super(part_weighter, self).__init__()
        self.w1 = nn.Linear(input_layer_dim, input_layer_dim//squeeze_scale)
        self.w2 = nn.Linear(input_layer_dim//squeeze_scale, input_layer_dim)

    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.w1(x)
        x = F.relu(x)
        x = self.w2(x)
        x = F.sigmoid(x)
        return x

class part_feature_extractor(nn.Module):
    def __init__(self, input_layer_dim, out_layer_dim):
        super(part_feature_extractor, self).__init__()
        self.w = nn.Linear(input_layer_dim, out_layer_dim)

    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.w(x)
        return x

class FGnet_1(nn.Module):
    def __init__(self, num_classes, loss, num_parts=2, part_feat_dim=128, **kwargs):
        super(FGnet_1, self).__init__()
        self.loss = loss
        self.part_weighters = nn.ModuleList([part_weighter(2048, 16) for i in range(num_parts)])
        self.part_feature_generators = nn.ModuleList([part_feature_extractor(2048*8*4, part_feat_dim) for i in range(num_parts)])
        self.classifier_global = nn.Linear(2048, num_classes)
        self.classifier_parts = nn.Linear(part_feat_dim*num_parts, num_classes)
        self._init_params()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.num_parts = num_parts

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

    def forward(self, x, return_featuremaps = False, return_partmaps = False):
        x = self.base(x)
        if return_featuremaps:
            return x

        f_global = F.avg_pool2d(x, x.size()[2:])
        f_global = f_global.view(f_global.size(0), -1)
        if not self.training and not return_partmaps:
            return f_global
        y_global = self.classifier_global(f_global)

        feature_parts = []
        b, c, w, h = x.size()
        part_maps_return = []
        for weighter, generator in zip(self.part_weighters, self.part_feature_generators):
            parts_weight = weighter(x)
            parts_weight = parts_weight.view(b, c, 1)
            x = x.view(b, c, -1)
            weighted_x = x * parts_weight
            weighted_x = weighted_x.view(b, c, w, h)
            if return_partmaps:
                part_maps_return.append([weighted_x, parts_weight])
            feat_part = generator(weighted_x)
            feature_parts.append(feat_part)
            x = x.view(b, c, w, h)
        if return_partmaps:
            return part_maps_return
        feature_parts = torch.cat(feature_parts, dim=1)
        y_parts = self.classifier_parts(feature_parts)

        if self.loss == 'softmax':
            return y_global, y_parts, part_maps_return
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class FGnet_1_resnet50m_fc_parts(nn.Module):
    def __init__(self, num_classes, loss, num_parts=2, part_feat_dim=1024, neck = False, last_stride = False, **kwargs):
        super(FGnet_1_resnet50m_fc_parts, self).__init__()
        self.loss = loss
        self.num_parts = num_parts
        self.part_weighters = nn.ModuleList([part_weighter(2048, 16) for i in range(num_parts)])
        self.part_feature_generators = nn.ModuleList([part_feature_extractor(2048, part_feat_dim) for i in range(num_parts)])
        self.classifier_parts = nn.Linear(part_feat_dim*num_parts, num_classes)
        self.classifier_global = nn.Linear(3072, num_classes)
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        if neck:
            self.bottleneck = nn.BatchNorm1d(3072)
            self.bottleneck.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck = None
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

    def forward(self, x, return_featuremaps = False, return_partmaps = False):
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
            feature_global = self.bottleneck(feature_t)
        else:
            feature_global = feature_t

        if not self.training and not return_partmaps:
            return feature_global

        y_global = self.classifier_global(feature_global)

        feature_parts = []
        b, c, w, h = x5c.size()
        part_maps_return = []
        for weighter, generator in zip(self.part_weighters, self.part_feature_generators):
            parts_weight = weighter(x5c)
            parts_weight = parts_weight.view(b, c, 1)
            x5c = x5c.view(b, c, -1)
            weighted_x = x5c * parts_weight
            weighted_x = weighted_x.view(b, c, w, h)
            part_maps_return.append([weighted_x, parts_weight])
            feat_part = generator(weighted_x)
            feature_parts.append(feat_part.unsqueeze(0))
            x5c = x5c.view(b, c, w, h)
        if return_partmaps:
            return part_maps_return
        feature_parts = torch.cat(feature_parts)
        feature_parts = feature_parts.permute(1, 0, 2)
        y_parts = self.classifier_parts(feature_parts.contiguous().view(b, -1))

        if self.loss == 'triplet_softmax':
            return y_global, y_parts, feature_global, part_maps_return
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class FGnet_1_resnet50m_fc_parts_bd(nn.Module):
    def __init__(self, num_classes, loss, num_parts=2, part_feat_dim=1024, neck = False, last_stride = False, drop_height_ratio=0.33, drop_width_ratio=1.0, **kwargs):
        super(FGnet_1_resnet50m_fc_parts_bd, self).__init__()
        self.loss = loss
        self.num_parts = num_parts
        self.part_weighters = nn.ModuleList([part_weighter(2048, 16) for i in range(num_parts)])
        self.part_feature_generators = nn.ModuleList([part_feature_extractor(2048, part_feat_dim) for i in range(num_parts)])
        self.classifier_parts = nn.Linear(part_feat_dim*num_parts, num_classes)
        self.classifier_global = nn.Linear(3072, num_classes)
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.batch_drop = BatchFeatureErase(num_classes, 2048, drop_height_ratio, drop_width_ratio)
        if neck:
            self.bottleneck = nn.BatchNorm1d(3072)
            self.bottleneck.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck = None
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

    def forward(self, x, return_featuremaps = False, return_partmaps = False):
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
            feature_global = self.bottleneck(feature_t)
        else:
            feature_global = feature_t

        if not self.training and not return_partmaps:
            return feature_global

        y_global = self.classifier_global(feature_global)

        feature_parts = []
        b, c, w, h = x5c.size()
        part_maps_return = []
        for weighter, generator in zip(self.part_weighters, self.part_feature_generators):
            parts_weight = weighter(x5c)
            parts_weight = parts_weight.view(b, c, 1)
            x5c = x5c.view(b, c, -1)
            weighted_x = x5c * parts_weight
            weighted_x = weighted_x.view(b, c, w, h)
            part_maps_return.append([weighted_x, parts_weight])
            feat_part = generator(weighted_x)
            feature_parts.append(feat_part.unsqueeze(0))
            x5c = x5c.view(b, c, w, h)
        if return_partmaps:
            return part_maps_return
        feature_parts = torch.cat(feature_parts)
        feature_parts = feature_parts.permute(1, 0, 2)
        y_parts = self.classifier_parts(feature_parts.contiguous().view(b, -1))

        if self.loss == 'triplet_softmax':
            return y_global, y_parts, feature_global, part_maps_return
        if self.loss == 'triplet_softmax_dropbatch':
            return y_global, y_parts, feature_global, part_maps_return, drop_features
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class FGnet_1_resnet50_fc_parts_bd(nn.Module):
    """
    """
    def __init__(self, num_classes=0, loss='softmax', num_parts=2, part_feat_dim=1024, neck = False, drop_height_ratio=0.33, drop_width_ratio=1.0, **kwargs):
        super(FGnet_1_resnet50_fc_parts_bd, self).__init__()
        self.loss = loss
        self.num_parts = num_parts
        #backbone
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

        #parts
        self.part_weighters = nn.ModuleList([part_weighter(2048, 16) for i in range(num_parts)])
        self.part_feature_generators = nn.ModuleList([part_feature_extractor(2048, part_feat_dim) for i in range(num_parts)])
        self.classifier_parts = nn.Linear(part_feat_dim*num_parts, num_classes)
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

    def forward(self, x, return_featuremaps = False, return_partmaps = False, return_gap_attention = False, drop_top=False):
        x = self.base(x)
        b, c, w, h = x.size()
        if return_featuremaps:
            return x
        drop_x = self.batch_drop(x, drop_top=drop_top)

        #global
        glob_x = self.avgpool(x)
        t_x = self.reduction_global(glob_x)
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
            if return_gap_attention:
                attention = self.get_activations(x).unsqueeze(1)
                if x.is_cuda: attention = attention.cuda()
                gap_attention = self.avgpool(x * attention).view(b, c)
                return torch.cat((x_x, x_drop_x, gap_attention), dim=1)
            else:
                return torch.cat((x_x, x_drop_x), dim=1)

        #parts
        feature_parts = []
        part_maps_return = []
        for weighter, generator in zip(self.part_weighters, self.part_feature_generators):
            parts_weight = weighter(x)
            parts_weight = parts_weight.view(b, c, 1)
            x = x.view(b, c, -1)
            weighted_x = x * parts_weight
            weighted_x = weighted_x.view(b, c, w, h)
            part_maps_return.append([weighted_x, parts_weight])
            feat_part = generator(weighted_x)
            feature_parts.append(feat_part.unsqueeze(0))
            x = x.view(b, c, w, h)
        if return_partmaps:
            return part_maps_return
        feature_parts = torch.cat(feature_parts)
        feature_parts = feature_parts.permute(1, 0, 2)
        x_parts_prelogits = self.classifier_parts(feature_parts.contiguous().view(b, -1))

        if self.loss == 'triplet_dropbatch':
            return x_prelogits, t_x, (x_drop_prelogits, t_drop_x)
        elif self.loss == 'triplet_softmax_dropbatch':
            return x_prelogits, x_parts_prelogits, t_x, part_maps_return, (x_drop_prelogits, t_drop_x)
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

    def get_activations(self, outputs, threshold = 0.35):
        outputs = (outputs**2).sum(1)
        b, h, w = outputs.size()
        outputs = outputs.view(b, h*w)
        outputs = F.normalize(outputs, p=2, dim=1)
        maxs, _ = torch.max(outputs, dim=1)
        mins, _ = torch.min(outputs, dim=1)
        maxs = maxs.view(b, 1)
        mins = mins.view(b, 1)
        outputs = 255 * (outputs - maxs) / (maxs - mins + 1e-12)
        outputs = torch.tensor(torch.floor(outputs).clone().detach(), dtype = torch.uint8)
        masks = torch.zeros_like(outputs, dtype = torch.float)
        masks[(outputs/255) >= threshold] = 1
        masks = masks.view(b, h, w)
        return masks


class FGnet_1_resnet50_fc_parts_bd_tri(nn.Module):
    """
    """
    def __init__(self, num_classes=0, loss='softmax', num_parts=2, part_feat_dim=1024, neck = False, drop_height_ratio=0.33, drop_width_ratio=1.0, **kwargs):
        super(FGnet_1_resnet50_fc_parts_bd_tri, self).__init__()
        self.loss = loss
        self.num_parts = num_parts
        #backbone
        if neck:
            self.bottleneck_global = nn.BatchNorm1d(512)
            self.bottleneck_global.bias.requires_grad_(False)  # no shift
            self.bottleneck_db = nn.BatchNorm1d(1024)
            self.bottleneck_db.bias.requires_grad_(False)  # no shift
            self.bottleneck_parts = nn.BatchNorm1d(part_feat_dim*num_parts)
            self.bottleneck_parts.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck_global = None
            self.bottleneck_db = None
            self.bottleneck_parts = None

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

        #parts
        self.part_weighters = nn.ModuleList([part_weighter(2048, 16) for i in range(num_parts)])
        self.part_feature_generators = nn.ModuleList([part_feature_extractor(2048, part_feat_dim) for i in range(num_parts)])
        self.classifier_parts = nn.Linear(part_feat_dim*num_parts, num_classes)
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

    def forward(self, x, return_featuremaps = False, return_partmaps = False, return_gap_attention = False, drop_top=False):
        x = self.base(x)
        b, c, w, h = x.size()
        if return_featuremaps:
            return x
        drop_x = self.batch_drop(x, drop_top=drop_top)

        #global
        glob_x = self.avgpool(x)
        t_x = self.reduction_global(glob_x)
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

        #parts
        feature_parts = []
        part_maps_return = []
        for weighter, generator in zip(self.part_weighters, self.part_feature_generators):
            parts_weight = weighter(x)
            parts_weight = parts_weight.view(b, c, 1)
            x = x.view(b, c, -1)
            weighted_x = x * parts_weight
            weighted_x = weighted_x.view(b, c, w, h)
            part_maps_return.append([weighted_x, parts_weight])
            feat_part = generator(weighted_x)
            feature_parts.append(feat_part.unsqueeze(0))
            x = x.view(b, c, w, h)
        if return_partmaps:
            return part_maps_return

        feature_parts = torch.cat(feature_parts)
        feature_parts = feature_parts.permute(1, 0, 2)
        t_parts_x = feature_parts.contiguous().view(b, -1)
        if self.bottleneck_parts:
            x_parts_x = self.bottleneck_parts(t_parts_x)
        else:
            x_parts_x = t_parts_x
        x_parts_prelogits = self.classifier_parts(x_parts_x)

        if self.loss == 'triplet_softmax_parts_dropbatch':
            return x_prelogits, x_parts_prelogits, t_x, t_parts_x, part_maps_return, (x_drop_prelogits, t_drop_x)
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class FGnet_2(nn.Module):
    def __init__(self, num_classes, loss, num_parts=2, part_feat_dim=128, last_stride=False, **kwargs):
        super(FGnet_2, self).__init__()
        self.loss = loss
        self.part_weighters = nn.ModuleList([part_weighter(2048, 16) for i in range(num_parts)])
        if last_stride:
            feature_size = 2048*16*8
        else:
            feature_size = 2048*8*4
        self.part_feature_generators = nn.ModuleList([part_feature_extractor(feature_size, part_feat_dim) for i in range(num_parts)])
        self.classifier_global = nn.Linear(2048, num_classes)
        self._init_params()

        if last_stride:
            resnet = resnet50_ls(num_classes, pretrained=True) #resnet50 with last stride = 1
        else:
            resnet = resnet50(num_classes, pretrained=True)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        self.num_parts = num_parts

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

    def forward(self, x, return_featuremaps = False, return_partmaps = False):
        x = self.base(x)
        if return_featuremaps:
            return x

        f_global = F.avg_pool2d(x, x.size()[2:])
        f_global = f_global.view(f_global.size(0), -1)
        if not self.training and not return_partmaps:
            return f_global
        y_global = self.classifier_global(f_global)

        feature_parts = []
        b, c, w, h = x.size()
        part_maps_return = []
        for weighter, generator in zip(self.part_weighters, self.part_feature_generators):
            parts_weight = weighter(x)
            parts_weight = parts_weight.view(b, c, 1)
            x = x.view(b, c, -1)
            weighted_x = x * parts_weight
            weighted_x = weighted_x.view(b, c, w, h)
            if return_partmaps:
                part_maps_return.append([weighted_x, parts_weight])
            feat_part = generator(weighted_x)
            feature_parts.append(feat_part.unsqueeze(0))
            x = x.view(b, c, w, h)
        if return_partmaps:
            return part_maps_return
        feature_parts = torch.cat(feature_parts)
        feature_parts = feature_parts.permute(1, 0, 2)

        if self.loss == 'softmax_npairs':
            return y_global, feature_parts
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class FGnet_2_resnet50m(nn.Module):
    def __init__(self, num_classes, loss, num_parts=2, part_feat_dim=1024, neck = False, last_stride = False, **kwargs):
        super(FGnet_2_resnet50m, self).__init__()
        self.loss = loss
        self.num_parts = num_parts
        self.part_weighters = nn.ModuleList([part_weighter(2048, 16) for i in range(num_parts)])
        if last_stride:
            feature_size = 2048*16*8
        else:
            feature_size = 2048*8*4
        self.part_feature_generators = nn.ModuleList([part_feature_extractor(2048, part_feat_dim) for i in range(num_parts)])
        self.classifier_global = nn.Linear(3072, num_classes)
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        if neck:
            self.bottleneck = nn.BatchNorm1d(3072)
            self.bottleneck.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck = None
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

    def forward(self, x, return_featuremaps = False, return_partmaps = False):
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
            feature_global = self.bottleneck(feature_t)
        else:
            feature_global = feature_t

        if not self.training and not return_partmaps:
            return feature_global

        y_global = self.classifier_global(feature_global)

        feature_parts = []
        b, c, w, h = x5c.size()
        part_maps_return = []
        for weighter, generator in zip(self.part_weighters, self.part_feature_generators):
            parts_weight = weighter(x5c)
            parts_weight = parts_weight.view(b, c, 1)
            x5c = x5c.view(b, c, -1)
            weighted_x = x5c * parts_weight
            weighted_x = weighted_x.view(b, c, w, h)
            part_maps_return.append([weighted_x, parts_weight])
            feat_part = generator(weighted_x)
            feature_parts.append(feat_part.unsqueeze(0))
            x5c = x5c.view(b, c, w, h)
        if return_partmaps:
            return part_maps_return
        feature_parts = torch.cat(feature_parts)
        feature_parts = feature_parts.permute(1, 0, 2)

        if self.loss == 'softmax_npairs':
            return y_global, feature_parts
        if self.loss == 'triplet_npairs':
            return y_global, feature_t, feature_parts, part_maps_return
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class FGnet_2_resnet50m_fc_parts(nn.Module):
    def __init__(self, num_classes, loss, num_parts=2, part_feat_dim=1024, neck = False, last_stride = False, **kwargs):
        super(FGnet_2_resnet50m_fc_parts, self).__init__()
        self.loss = loss
        self.num_parts = num_parts
        self.part_weighters = nn.ModuleList([part_weighter(2048, 16) for i in range(num_parts)])
        if last_stride:
            feature_size = 2048*16*8
        else:
            feature_size = 2048*8*4
        self.part_feature_generators = nn.ModuleList([part_feature_extractor(2048, part_feat_dim) for i in range(num_parts)])
        self.classifier_parts = nn.Linear(part_feat_dim*num_parts, num_classes)
        self.classifier_global = nn.Linear(3072, num_classes)
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        if neck:
            self.bottleneck = nn.BatchNorm1d(3072)
            self.bottleneck.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck = None
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

    def forward(self, x, return_featuremaps = False, return_partmaps = False):
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
            feature_global = self.bottleneck(feature_t)
        else:
            feature_global = feature_t

        if not self.training and not return_partmaps:
            return feature_global

        y_global = self.classifier_global(feature_global)

        feature_parts = []
        b, c, w, h = x5c.size()
        part_maps_return = []
        for weighter, generator in zip(self.part_weighters, self.part_feature_generators):
            parts_weight = weighter(x5c)
            parts_weight = parts_weight.view(b, c, 1)
            x5c = x5c.view(b, c, -1)
            weighted_x = x5c * parts_weight
            weighted_x = weighted_x.view(b, c, w, h)
            part_maps_return.append([weighted_x, parts_weight])
            feat_part = generator(weighted_x)
            feature_parts.append(feat_part.unsqueeze(0))
            x5c = x5c.view(b, c, w, h)
        if return_partmaps:
            return part_maps_return
        feature_parts = torch.cat(feature_parts)
        feature_parts = feature_parts.permute(1, 0, 2)
        y_parts = self.classifier_parts(feature_parts.contiguous().view(b, -1))

        if self.loss == 'softmax_npairs':
            return y_global, feature_parts
        if self.loss == 'triplet_npairs':
            return y_global, feature_t, feature_parts, part_maps_return
        if self.loss == 'triplet_npairs_softmax' or self.loss == 'triplet_npairs_softmax_separate':
            return y_global, feature_t, feature_parts, part_maps_return, y_parts
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

#batch dropblock

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

class BatchDropTop(nn.Module):
    def __init__(self, h_ratio):
        super(BatchDropTop, self).__init__()
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
        self.drop_batch_drop_top = BatchDropTop(h_ratio)

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

class BatchFeatureErase(nn.Module):
    def __init__(self, num_classes, channels, h_ratio=0.33, w_ratio=1.):
        super(BatchFeatureErase, self).__init__()
        self.drop_batch_bottleneck = Bottleneck(channels, 512)
        self.drop_batch_drop = BatchDrop(h_ratio, w_ratio)
        self.drop_part_maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.drop_batch_reduction = nn.Sequential(
            nn.Linear(channels, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        x = self.drop_batch_bottleneck(x)
        x = self.drop_batch_drop(x)
        x = self.drop_part_maxpool(x).view(x.size()[:2])
        x = self.drop_batch_reduction(x)
        prelogits = self.classifier(x)
        return (prelogits, x)

class FGnet_2_resnet50m_fc_parts_bd(nn.Module):
    def __init__(self, num_classes, loss, num_parts=2, part_feat_dim=128, neck = False, last_stride = False, drop_height_ratio=0.33, drop_width_ratio=1.0, **kwargs):
        super(FGnet_2_resnet50m_fc_parts_bd, self).__init__()
        self.loss = loss
        self.num_parts = num_parts
        self.part_weighters = nn.ModuleList([part_weighter(2048, 16) for i in range(num_parts)])
        if last_stride:
            feature_size = 2048*16*8
        else:
            feature_size = 2048*8*4
        self.part_feature_generators = nn.ModuleList([part_feature_extractor(feature_size, part_feat_dim) for i in range(num_parts)])
        self.classifier_global = nn.Linear(3072, num_classes)
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        if neck:
            self.bottleneck = nn.BatchNorm1d(3072)
            self.bottleneck.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck = None
        #drop batch
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

    def forward(self, x, return_featuremaps = False, return_partmaps = False):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        if return_featuremaps:
            return x5c

        #dropbatch features
        drop_features = self.batch_drop(x5c)

        #global 50m features
        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        feature_t = torch.cat((x5c_feat, midfeat), dim=1)
        if self.bottleneck:
            feature_global = self.bottleneck(feature_t)
        else:
            feature_global = feature_t

        if not self.training and not return_partmaps:
            return feature_global

        y_global = self.classifier_global(feature_global)

        #parts features
        feature_parts = []
        b, c, w, h = x5c.size()
        part_maps_return = []
        for weighter, generator in zip(self.part_weighters, self.part_feature_generators):
            parts_weight = weighter(x5c)
            parts_weight = parts_weight.view(b, c, 1)
            x5c = x5c.view(b, c, -1)
            weighted_x = x5c * parts_weight
            weighted_x = weighted_x.view(b, c, w, h)
            if return_partmaps:
                part_maps_return.append([weighted_x, parts_weight])
            feat_part = generator(weighted_x)
            feature_parts.append(feat_part.unsqueeze(0))
            x5c = x5c.view(b, c, w, h)
        if return_partmaps:
            return part_maps_return
        feature_parts = torch.cat(feature_parts)
        feature_parts = feature_parts.permute(1, 0, 2)

        if self.loss == 'triplet_npairs_dropbatch':
            return y_global, feature_t, feature_parts, drop_features
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

## FG net

def fgnet_1_1_resnet50_fc_parts_bd_tri(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1_resnet50_fc_parts_bd_tri(
        num_classes=num_classes,
        loss=loss,
        num_parts = 1,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model

def fgnet_1_1_resnet50_neck_fc_parts_bd_tri(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1_resnet50_fc_parts_bd_tri(
        num_classes=num_classes,
        loss=loss,
        num_parts = 1,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model

def fgnet_1_1_resnet50_fc_parts_bd(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1_resnet50_fc_parts_bd(
        num_classes=num_classes,
        loss=loss,
        num_parts = 1,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model

def fgnet_1_2_resnet50_fc_parts_bd(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1_resnet50_fc_parts_bd(
        num_classes=num_classes,
        loss=loss,
        num_parts = 2,
        neck=False,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model

def fgnet_1_1_resnet50_neck_fc_parts_bd(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1_resnet50_fc_parts_bd(
        num_classes=num_classes,
        loss=loss,
        num_parts = 1,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model

def fgnet_1_2_resnet50_neck_fc_parts_bd(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1_resnet50_fc_parts_bd(
        num_classes=num_classes,
        loss=loss,
        num_parts = 2,
        neck=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model

def fgnet_1_2(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1(
        num_classes=num_classes,
        loss=loss,
        num_parts = 2,
        **kwargs
    )
    return model

def fgnet_1_1_resnet50m_ls_neck_fc_parts(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1_resnet50m_fc_parts(
        num_classes=num_classes,
        loss=loss,
        num_parts = 1,
        neck=True,
        last_stride=True,
        **kwargs
    )
    return model

def fgnet_1_1_resnet50m_ls_neck_fc_parts_bd(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1_resnet50m_fc_parts_bd(
        num_classes=num_classes,
        loss=loss,
        num_parts = 1,
        neck=True,
        last_stride=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model

def fgnet_1_2_resnet50m_ls_neck_fc_parts_bd(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1_resnet50m_fc_parts_bd(
        num_classes=num_classes,
        loss=loss,
        num_parts = 2,
        neck=True,
        last_stride=True,
        drop_height_ratio=0.33,
        drop_width_ratio=1.0,
        **kwargs
    )
    return model

def fgnet_1_2_resnet50m_ls_neck_fc_parts(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1_resnet50m_fc_parts(
        num_classes=num_classes,
        loss=loss,
        num_parts = 2,
        neck=True,
        last_stride=True,
        **kwargs
    )
    return model

def fgnet_1_3_resnet50m_ls_neck_fc_parts(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1_resnet50m_fc_parts(
        num_classes=num_classes,
        loss=loss,
        num_parts = 3,
        neck=True,
        last_stride=True,
        **kwargs
    )
    return model

def fgnet_2_2(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2(
        num_classes=num_classes,
        loss=loss,
        num_parts = 2,
        last_stride=False,
        **kwargs
    )
    return model

def fgnet_2_2_ls_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2(
        num_classes=num_classes,
        loss=loss,
        num_parts = 2,
        last_stride=True,
        neck=True,
        **kwargs
    )
    return model

def fgnet_2_2_resnet50m(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2_resnet50m(
        num_classes=num_classes,
        loss=loss,
        num_parts = 2,
        neck=False,
        **kwargs
    )
    return model

def fgnet_2_2_resnet50m_ls_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2_resnet50m(
        num_classes=num_classes,
        loss=loss,
        num_parts = 2,
        neck=True,
        last_stride=True,
        **kwargs
    )
    return model

def fgnet_2_2_resnet50m_ls_neck_fc_parts(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2_resnet50m_fc_parts(
        num_classes=num_classes,
        loss=loss,
        num_parts = 2,
        neck=True,
        last_stride=True,
        **kwargs
    )
    return model

def fgnet_2_2_resnet50m_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2_resnet50m(
        num_classes=num_classes,
        loss=loss,
        num_parts = 2,
        neck=True,
        **kwargs
    )
    return model

def fgnet_1_3(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1(
        num_classes=num_classes,
        loss=loss,
        num_parts = 3,
        **kwargs
    )
    return model

def fgnet_2_3(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2(
        num_classes=num_classes,
        loss=loss,
        num_parts = 3,
        **kwargs
    )
    return model

def fgnet_2_3_resnet50m(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2_resnet50m(
        num_classes=num_classes,
        loss=loss,
        num_parts = 3,
        neck=False,
        **kwargs
    )
    return model

def fgnet_2_3_resnet50m_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2_resnet50m(
        num_classes=num_classes,
        loss=loss,
        num_parts = 3,
        neck=True,
        **kwargs
    )
    return model


def fgnet_2_3_resnet50m_ls_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2_resnet50m(
        num_classes=num_classes,
        loss=loss,
        num_parts = 3,
        neck=True,
        last_stride=True,
        **kwargs
    )
    return model

def fgnet_2_3_resnet50m_ls_neck_fc_parts(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2_resnet50m_fc_parts(
        num_classes=num_classes,
        loss=loss,
        num_parts = 3,
        neck=True,
        last_stride=True,
        **kwargs
    )
    return model

def fgnet_2_4_resnet50m_ls_neck_fc_parts(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2_resnet50m_fc_parts(
        num_classes=num_classes,
        loss=loss,
        num_parts = 4,
        neck=True,
        last_stride=True,
        **kwargs
    )
    return model

def fgnet_2_3_ls_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2(
        num_classes=num_classes,
        loss=loss,
        num_parts = 3,
        last_stride=True,
        neck=True,
        **kwargs
    )
    return model

def fgnet_1_4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_1(
        num_classes=num_classes,
        loss=loss,
        num_parts = 4,
        **kwargs
    )
    return model

def fgnet_2_4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2(
        num_classes=num_classes,
        loss=loss,
        num_parts = 4,
        **kwargs
    )
    return model

def fgnet_2_4_resnet50m(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2_resnet50m(
        num_classes=num_classes,
        loss=loss,
        num_parts = 4,
        neck=False,
        **kwargs
    )
    return model

def fgnet_2_4_resnet50m_neck(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2_resnet50m(
        num_classes=num_classes,
        loss=loss,
        num_parts = 4,
        neck=True,
        **kwargs
    )
    return model

def fgnet_2_5(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_2(
        num_classes=num_classes,
        loss=loss,
        num_parts = 5,
        **kwargs
    )
    return model

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=False, activation = 'leakyrelu', dropout = False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope = 0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Not a valid activation, received {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class Deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=False, activation = 'leakyrelu', dropout = False):
        super(Deconv2d, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope = 0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Not a valid activation, received {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class attention_autoencoder(nn.Module):
    def __init__(self):
        super(attention_autoencoder, self).__init__()
        self.encoder_1 = Conv2d(64, 64, 6, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
        self.encoder_2 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
        self.encoder_3 = Conv2d(64, 64, 4, stride = 2, bn = False, activation = 'leakyrelu', dropout=False)
        self.encoder_4 = Conv2d(64, 64, 3, stride = 1, bn = False, activation = 'leakyrelu', dropout=False)

        self.decoder_1 = Deconv2d(64, 64, 3, stride = 1, bn = False, activation = 'relu', dropout = True)
        self.decoder_2 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = True)
        self.decoder_3 = Deconv2d(128, 64, 4, stride = 2, bn = False, activation = 'relu', dropout = False)
        self.decoder_4 = Deconv2d(128, 1, 6, stride = 2, bn = False, activation = 'relu', dropout = False)

    def forward(self, x):
        e1 = self.encoder_1(x)
        e2 = self.encoder_2(e1)
        e3 = self.encoder_3(e2)
        e4 = self.encoder_4(e3)

        d = self.decoder_1(e4)
        d = torch.cat((d, e3), dim=1)
        d = self.decoder_2(d)
        d = torch.cat((d, e2), dim=1)
        d = self.decoder_3(d)
        d = torch.cat((d, e1), dim=1)
        d = self.decoder_4(d)
        return d

class FGnet_3_resnet50m_fc_parts(nn.Module):
    def __init__(self, num_classes, loss, num_parts=2, part_feat_dim=1024, neck = False, last_stride = False, **kwargs):
        super(FGnet_3_resnet50m_fc_parts, self).__init__()
        self.loss = loss
        self.num_parts = num_parts
        self.channel_part_attentions = nn.ModuleList([part_weighter(2048, 16) for i in range(num_parts)])
        self.spatial_part_attentions = nn.ModuleList([attention_autoencoder() for i in range(num_parts)])
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.part_feature_generators = nn.ModuleList([part_feature_extractor(2048, part_feat_dim) for i in range(num_parts)])
        self.classifier_parts = nn.Linear(part_feat_dim*num_parts, num_classes)
        self.classifier_global = nn.Linear(3072, num_classes)
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        if neck:
            self.bottleneck = nn.BatchNorm1d(3072)
            self.bottleneck.bias.requires_grad_(False)  # no shift
        else:
            self.bottleneck = None
        self.attention_join = nn.Sequential(
            nn.Conv2d(2048, 2048, 1),
            nn.Sigmoid()
        )
        self.spatial_attention_head = nn.Sequential(
            nn.Conv2d(2048, 64, 1),
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )
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

    def forward(self, x, return_featuremaps = False, return_partmaps = False):
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
            feature_global = self.bottleneck(feature_t)
        else:
            feature_global = feature_t

        if not self.training and not return_partmaps:
            return feature_global

        y_global = self.classifier_global(feature_global)

        feature_parts = []
        b, c, w, h = x5c.size()
        attention_maps = []
        for channel_attention, spatial_attention, generator in zip(self.channel_part_attentions, self.spatial_part_attentions, self.part_feature_generators):
            channel_weight = channel_attention(x5c).view(b, c, 1, 1)
            spatial_weight = spatial_attention(self.spatial_attention_head(x5c)).view(b, 1, w, h)
            tensor_attention = channel_weight * spatial_weight
            tensor_attention = self.attention_join(tensor_attention)
            weighted_x = x5c * (1 + tensor_attention)
            attention_maps.append([x5c * tensor_attention, channel_weight, spatial_weight, tensor_attention])
            feat_part = generator(weighted_x)
            feature_parts.append(feat_part.unsqueeze(0))
            x5c = x5c.view(b, c, w, h)
        if return_partmaps:
            return attention_maps
        feature_parts = torch.cat(feature_parts)
        feature_parts = feature_parts.permute(1, 0, 2)
        y_parts = self.classifier_parts(feature_parts.contiguous().view(b, -1))

        if self.loss == 'triplet_npairs_softmax_csattention':
            return y_global, feature_t, y_parts, feature_parts, attention_maps
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def fgnet_3_2_resnet50m_ls_neck_fc_parts(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_3_resnet50m_fc_parts(
        num_classes=num_classes,
        loss=loss,
        num_parts = 2,
        neck=True,
        last_stride=True,
        **kwargs
    )
    return model

def fgnet_3_3_resnet50m_ls_neck_fc_parts(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = FGnet_3_resnet50m_fc_parts(
        num_classes=num_classes,
        loss=loss,
        num_parts = 3,
        neck=True,
        last_stride=True,
        **kwargs
    )
    return model
