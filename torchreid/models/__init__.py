from __future__ import absolute_import

import torch

from .resnet import *
from .resnetmid import *
from .senet import *
from .densenet import *
from .inceptionresnetv2 import *
from .inceptionv4 import *
from .xception import *
from .msrn import *

from .nasnet import *
from .mobilenetv2 import *
from .shufflenet import *
from .squeezenet import *
from .shufflenetv2 import *

from .mudeep import *
from .hacnn import *
from .pcb import *
from .mlfn import *
from .osnet import *
from .fgnet import *


__model_factory = {
    # image classification models
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet50_bd': resnet50_bd,
    'resnet50_bd_neck': resnet50_bd_neck,
    'resnet50_bd_botdropfeat': resnet50_bd_botdropfeat, 
    'resnet50_bd_neck_botdropfeat': resnet50_bd_neck_botdropfeat, 
    'resnet50_bd_botdropfeat_doubot': resnet50_bd_botdropfeat_doubot, 
    'resnet50_bd_neck_botdropfeat_doubot': resnet50_bd_neck_botdropfeat_doubot,
    'resnet50_bd_neck_botdropfeat_doubot_pose': resnet50_bd_neck_botdropfeat_doubot_pose,
    'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_1': resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_1,
    'resnet50_bd_neck_botdropfeat_doubot_pose_13_conv_1': resnet50_bd_neck_botdropfeat_doubot_pose_13_conv_1,
    'resnet50_bd_neck_botdropfeat_doubot_pose_13_conv_2': resnet50_bd_neck_botdropfeat_doubot_pose_13_conv_2,
    'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_4': resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_4,
    'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_5': resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_5,
    'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_6': resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_6,
    'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_7': resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_7,
    'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_8': resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_8,
    'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_9': resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_9,
    'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_10': resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_10,
    'resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_11': resnet50_bd_neck_botdropfeat_doubot_pose_13_fc_11,
    'resnet50_bd_botdropfeat_inter': resnet50_bd_botdropfeat_inter,
    'resnet50_bd_neck_botdropfeat_inter': resnet50_bd_neck_botdropfeat_inter,
    'resnet50_bd_botdropfeat_doubot_inter': resnet50_bd_botdropfeat_doubot_inter,
    'resnet50_bd_neck_botdropfeat_doubot_inter': resnet50_bd_neck_botdropfeat_doubot_inter,
    'resnet50_ls': resnet50_ls,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet50m': resnet50m,
    'resnet50m_neck': resnet50m_neck,
    'resnet50m_ls_neck': resnet50m_ls_neck,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnet50_fc512': resnet50_fc512,
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet161': densenet161,
    'densenet121_fc512': densenet121_fc512,
    'inceptionresnetv2': inceptionresnetv2,
    'inceptionv4': inceptionv4,
    'xception': xception,
    # lightweight models
    'nasnsetmobile': nasnetamobile,
    'mobilenetv2_x1_0': mobilenetv2_x1_0,
    'mobilenetv2_x1_4': mobilenetv2_x1_4,
    'shufflenet': shufflenet,
    'squeezenet1_0': squeezenet1_0,
    'squeezenet1_0_fc512': squeezenet1_0_fc512,
    'squeezenet1_1': squeezenet1_1,
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': shufflenet_v2_x2_0,
    # reid-specific models
    'mudeep': MuDeep,
    'resnet50mid': resnet50mid,
    'hacnn': HACNN,
    'pcb_p6': pcb_p6,
    'pcb_p4': pcb_p4,
    'mlfn': mlfn,
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_x0_5': osnet_x0_5,
    'osnet_x0_25': osnet_x0_25,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
    'fgnet_1_2': fgnet_1_2,
    'fgnet_1_3': fgnet_1_3,
    'fgnet_1_4': fgnet_1_4,
    'fgnet_2_2': fgnet_2_2,
    'fgnet_2_3': fgnet_2_3,
    'fgnet_2_4': fgnet_2_4,
    'fgnet_2_5': fgnet_2_5,
    'fgnet_2_2_resnet50m': fgnet_2_2_resnet50m,
    'fgnet_2_3_resnet50m': fgnet_2_3_resnet50m,
    'fgnet_2_4_resnet50m': fgnet_2_4_resnet50m,
    'fgnet_2_2_resnet50m_neck': fgnet_2_2_resnet50m_neck,
    'fgnet_2_3_resnet50m_neck': fgnet_2_3_resnet50m_neck,
    'fgnet_2_4_resnet50m_neck': fgnet_2_4_resnet50m_neck,
    'fgnet_2_2_ls_neck':fgnet_2_2_ls_neck,
    'fgnet_2_3_ls_neck':fgnet_2_3_ls_neck,
    'fgnet_2_2_resnet50m_ls_neck': fgnet_2_2_resnet50m_ls_neck,
    'fgnet_2_3_resnet50m_ls_neck': fgnet_2_3_resnet50m_ls_neck,
    'resnet50m_bd_ls_neck': resnet50m_bd_ls_neck,
    'fgnet_2_2_resnet50m_ls_neck_fc_parts':fgnet_2_2_resnet50m_ls_neck_fc_parts,
    'fgnet_2_3_resnet50m_ls_neck_fc_parts':fgnet_2_3_resnet50m_ls_neck_fc_parts,
    'fgnet_2_4_resnet50m_ls_neck_fc_parts':fgnet_2_4_resnet50m_ls_neck_fc_parts,
    'fgnet_1_1_resnet50m_ls_neck_fc_parts':fgnet_1_1_resnet50m_ls_neck_fc_parts,
    'fgnet_1_2_resnet50m_ls_neck_fc_parts':fgnet_1_2_resnet50m_ls_neck_fc_parts,
    'fgnet_1_3_resnet50m_ls_neck_fc_parts':fgnet_1_3_resnet50m_ls_neck_fc_parts,
    'fgnet_1_1_resnet50m_ls_neck_fc_parts_bd':fgnet_1_1_resnet50m_ls_neck_fc_parts_bd,
    'fgnet_1_2_resnet50m_ls_neck_fc_parts_bd':fgnet_1_2_resnet50m_ls_neck_fc_parts_bd,
    'fgnet_3_2_resnet50m_ls_neck_fc_parts':fgnet_3_2_resnet50m_ls_neck_fc_parts,
    'fgnet_3_3_resnet50m_ls_neck_fc_parts':fgnet_3_3_resnet50m_ls_neck_fc_parts,
    'fgnet_1_1_resnet50_fc_parts_bd': fgnet_1_1_resnet50_fc_parts_bd,
    'fgnet_1_2_resnet50_fc_parts_bd': fgnet_1_2_resnet50_fc_parts_bd,
    'fgnet_1_1_resnet50_neck_fc_parts_bd': fgnet_1_1_resnet50_neck_fc_parts_bd,
    'fgnet_1_2_resnet50_neck_fc_parts_bd': fgnet_1_2_resnet50_neck_fc_parts_bd,
    'fgnet_1_1_resnet50_fc_parts_bd_tri': fgnet_1_1_resnet50_fc_parts_bd_tri, 
    'fgnet_1_1_resnet50_neck_fc_parts_bd_tri': fgnet_1_1_resnet50_neck_fc_parts_bd_tri,
    #MSRN
    'msrn_bd_botdropfeat_doubot':msrn_bd_botdropfeat_doubot,
    'msrn_bd_neck_botdropfeat_doubot':msrn_bd_neck_botdropfeat_doubot,
}


def show_avai_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def build_model(name, num_classes, loss='softmax', pretrained=True, use_gpu=True, batch_num_classes = -1):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        batch_num_classes=batch_num_classes
    )