from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class GroupLoss(nn.Module):
    """group loss
    
    Reference:
        Deeply Associative Two-stage Representations Learning based on Labels Interval Extension loss and Group loss for Person Re-identificatio

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    
    def __init__(self):
        super(GroupLoss, self).__init__()

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
        uniques = torch.unique(targets)
        intra = []
        for elem in uniques:
            mask = (targets == elem)
            mask = mask.expand(n, n) * (mask.expand(n, n).t())
            dist_elem = dist[mask]
            nn = mask.size(0)
            dist_elem = (torch.sum(dist_elem))/(nn*(nn-1))
            intra.append(dist_elem.unsqueeze(0))
        extra = []
        for i, d in enumerate(intra):
            sum = 0
            for j, dd in enumerate(intra):
                if i != j:
                    sum += torch.abs(d - dd)
            sum /= len(intra) - 1
            extra.append(sum.unsqueeze(0))
        extra = torch.cat(extra)
        intra = torch.cat(intra)
        extra = extra.clamp(min=1e-12)
        return torch.mean(intra + 1/extra)