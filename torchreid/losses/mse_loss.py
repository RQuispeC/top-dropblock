from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
        """
        if torch.is_tensor(inputs) and torch.is_tensor(targets): return self.mse_loss(inputs, targets)
        else: return torch.tensor(0)