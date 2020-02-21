from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""Focal Loss.
    
    Reference:
        Tsung-Yi et al. Focal Loss for Dense Object Detection.
        Cheng et al. Unified Multifaceted Feature Learning for Person Re-Identificatio
    Args:
        num_classes (int): number of classes.
        epsilon (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, gamma=2, size_average=True, use_gpu=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        targets = targets.view(-1,1)
        logpt = self.logsoftmax(inputs)
        logpt = torch.gather(logpt, dim=1, index=targets)
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)
        if self.use_gpu: pt = pt.cuda()
        loss = -1. * ((1. - pt)**self.gamma) * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
