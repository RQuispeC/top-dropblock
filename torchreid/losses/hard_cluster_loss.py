from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class ClusterLoss(nn.Module):
    """
    this groups aims to minimize the distance between the nearests and farthest element (with respect to an anchor) in a cluster
    Args:
    """
    
    def __init__(self):
        super(ClusterLoss, self).__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=0.0)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap_nearest, dist_ap_farthest = [], []
        for i in range(n):
            far = dist[i][mask[i]].max()
            near = torch.topk(dist[i][mask[i]], k=2, largest=False)[0][1]#ignore indices and distance to i-th anchor itself
            dist_ap_farthest.append(far.unsqueeze(0))
            dist_ap_nearest.append(near.unsqueeze(0))
        dist_ap_farthest = torch.cat(dist_ap_farthest)
        dist_ap_nearest = torch.cat(dist_ap_nearest)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_ap_nearest)
        return self.ranking_loss(dist_ap_nearest, dist_ap_farthest, y)
