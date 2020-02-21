from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class NPairsLoss(nn.Module):
    """N-pairs loss as explained in explained in equation 11 of MAMC paper.
    
    Reference:
        Multi-Attention Multi-Class Constraint for Fine-grained Image Recognition
    """
    
    def __init__(self, margin_sasc=0.3, margin_sadc=0.2, margin_dasc=0.4, use_gpu=True):
        super(NPairsLoss, self).__init__()
        self.use_gpu = use_gpu
        self.margin_sasc = margin_sasc
        self.margin_sadc = margin_sadc
        self.margin_dasc = margin_dasc
        self.ranking_loss_sasc = nn.MarginRankingLoss(margin=margin_sasc)
        self.ranking_loss_sadc = nn.MarginRankingLoss(margin=margin_sadc)
        self.ranking_loss_dasc = nn.MarginRankingLoss(margin=margin_dasc)


    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, part_num, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        b, p, _ = inputs.size()
        n = b * p
        inputs = inputs.contiguous().view(n, -1)
        targets = torch.repeat_interleave(targets, p)
        parts = torch.arange(p).repeat(b)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()
        if self.use_gpu: parts = parts.cuda()

        same_class_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        same_atten_mask = parts.expand(n, n).eq(parts.expand(n, n).t())

        s_sasc = same_class_mask & same_atten_mask
        s_sadc = (~same_class_mask) & same_atten_mask
        s_dasc = same_class_mask & (~same_atten_mask)
        s_dadc = (~same_class_mask) & (~same_atten_mask)

        # For each anchor, compute equation (11) of paper
        dist_ap_sasc, dist_an_sasc = [], []
        dist_ap_sadc, dist_an_sadc = [], []
        dist_ap_dasc, dist_an_dasc = [], []
        for i in range(n):
            #loss_sasc
            pos = dist[i][s_sasc[i]]
            neg = dist[i][s_sadc[i] | s_dasc[i] | s_dadc[i]]
            dist_ap_sasc.append(pos.max().unsqueeze(0))
            dist_an_sasc.append(neg.min().unsqueeze(0))

            #loss_sadc
            pos = dist[i][s_sadc[i]]
            neg = dist[i][s_dadc[i]]
            dist_ap_sadc.append(pos.max().unsqueeze(0))
            dist_an_sadc.append(neg.min().unsqueeze(0))

            #loss_dasc
            pos = dist[i][s_dasc[i]]
            neg = dist[i][s_dadc[i]]
            dist_ap_dasc.append(pos.max().unsqueeze(0))
            dist_an_dasc.append(neg.min().unsqueeze(0))

        dist_ap_sasc = torch.cat(dist_ap_sasc)
        dist_an_sasc = torch.cat(dist_an_sasc)
        dist_ap_sadc = torch.cat(dist_ap_sadc)
        dist_an_sadc = torch.cat(dist_an_sadc)
        dist_ap_dasc = torch.cat(dist_ap_dasc)
        dist_an_dasc = torch.cat(dist_an_dasc)

        y_sasc = torch.ones_like(dist_ap_sasc)
        y_sadc = torch.ones_like(dist_ap_sadc)
        y_dasc = torch.ones_like(dist_ap_dasc)
        loss_sasc = self.ranking_loss_sasc(dist_an_sasc, dist_ap_sasc, y_sasc)
        loss_sadc = self.ranking_loss_sadc(dist_an_sadc, dist_ap_sadc, y_sadc)
        loss_dasc = self.ranking_loss_dasc(dist_an_dasc, dist_ap_dasc, y_dasc)
        #print(loss_sasc + loss_sadc + loss_dasc)
        return loss_sasc + loss_sadc + loss_dasc

#http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
#https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py
#https://github.com/ChaofWang/Npair_loss_pytorch/blob/master/Npair_loss.py
#https://pytorch.org/docs/stable/nn.html#crossentropyloss
#https://kobiso.github.io/research/research-n-pair-loss/