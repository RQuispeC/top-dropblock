from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class ReferenceLoss(nn.Module):
    """Triplet loss with hard positive/negative mining + reference loss

    Reference:
        Reference-oriented Loss for Person Re-identification
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    
    def __init__(self, margin=0.3):
        super(ReferenceLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

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
        dist_ap, dist_an, dist_ref = [], [], []
        for i in range(n):
            pos_sort = torch.argsort(dist[i][mask[i]])
            neg_sort = torch.argsort(dist[i][mask[i] == 0])
            j = pos_sort[-1]
            k = neg_sort[0]
            dist_ap.append(dist[i][j].unsqueeze(0))
            dist_an.append(dist[i][k].unsqueeze(0))
            dist_ref.append(torch.abs(dist[i][k] - dist[j][k]).unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        dist_ref = torch.cat(dist_ref)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y) + torch.mean(dist_ref)