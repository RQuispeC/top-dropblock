from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from torch.nn import functional as F

class SeparationLoss(nn.Module):
    """
    """
    
    def __init__(self, use_gpu=True):
        super(SeparationLoss, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, part_num, c, h, w).
            targets (torch.Tensor): not used in this function
        """
        b, p, c, h, w = inputs.size()
        masks = torch.sigmoid(inputs.sum(2))
        #masks[masks >= 0.5] = 1
        #masks[masks < 0.5] = 0
        inputs = (inputs**2).sum(2)
        inputs = inputs.view(b, p, h*w)
        inputs = F.normalize(inputs, p=2, dim=2)
        inputs = inputs.view(b, p, h, w)
        loss = 0
        for i in range(b):
            den = torch.sum(inputs[i])
            den = den.clamp(min=1e-12)
            num = 0
            for j in range(p):
                for k in range(j+1, p):
                    num += torch.sum(torch.min(inputs[i, j], inputs[i, k]))
                    #num += torch.sum(torch.min(inputs[i, j], inputs[i, k]) * masks[i, j])
                    #num += torch.sum(torch.min(inputs[i, j], inputs[i, k]) * masks[i, k])
            loss += num/den
        return loss