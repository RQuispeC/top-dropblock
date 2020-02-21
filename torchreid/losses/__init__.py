from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss
from .hard_mine_triplet_loss_dependency import TripletLossWithDependency
from .npairs_loss import NPairsLoss
from .separation_loss import SeparationLoss
from .reference_loss import ReferenceLoss
from .group_loss import GroupLoss
from .focal_loss import FocalLoss
from .hard_cluster_loss import ClusterLoss
from .mse_loss import MSELoss

def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss