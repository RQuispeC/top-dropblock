from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np

class TripletLossWithDependency(nn.Module):
    """Triplet loss with hard positive/negative mining using  dependency between various distribuitions of same data
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLossWithDependency, self).__init__()
        self.margin = margin
        self.ranking_loss_ap_bn = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_an_bp = nn.MarginRankingLoss(margin=margin)

    def get_distance(self, inputs):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def forward(self, inputs_a_b, targets):
        """
        Args:
            inputs_a_b (tuple of torch.Tensor): tuple of feature matrix with shape (batch_size, feat_dim_a) and (batch_size, feat_dim_b).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        (inputs_a, inputs_b) = inputs_a_b
        n = inputs_a.size(0)

        a_len = inputs_a.size(1)
        b_len = inputs_b.size(1)
        lcm = np.lcm(a_len, b_len)

        # Compute pairwise distance, replace by the official when merged
        dist_a = self.get_distance(inputs_a)*(lcm/a_len)
        dist_b = self.get_distance(inputs_b)*(lcm/b_len)
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        dist_bp, dist_bn = [], []
        for i in range(n):
            dist_ap.append(dist_a[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist_a[i][mask[i] == 0].min().unsqueeze(0))
            dist_bp.append(dist_b[i][mask[i]].max().unsqueeze(0))
            dist_bn.append(dist_b[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        dist_bp = torch.cat(dist_bp)
        dist_bn = torch.cat(dist_bn)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_ap)
        return self.ranking_loss_ap_bn(dist_bn, dist_ap, y) + self.ranking_loss_an_bp(dist_an, dist_bp, y)