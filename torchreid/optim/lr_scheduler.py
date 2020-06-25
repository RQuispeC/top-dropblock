from __future__ import absolute_import
from __future__ import print_function

import torch


AVAI_SCH = ['single_step', 'multi_step', 'cosine', 'warmup_db', 'warmup_sb']

def warmup_db(ep, stepsize):
    ep +=1
    if ep < stepsize[0]:
        lr = 1e-4*(ep//5+1)
    elif ep < stepsize[1]:
        lr = 1e-3
    elif ep < stepsize[2]:
        lr = 1e-4
    else:
        lr = 1e-5
    lr /= 2
    return lr

def warmup_sb(ep, stepsize):
    ep +=1
    if ep < stepsize[0]:
        lr = 3.5e-5*(ep//10+1)
    elif ep < stepsize[1]:
        lr = 3.5e-4
    elif ep < stepsize[2]:
        lr = 3.5e-5
    else:
        lr = 3.5e-6
    lr /= 2
    return lr

def build_lr_scheduler(optimizer, lr_scheduler='single_step', stepsize=1, gamma=0.1, max_epoch=1):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str, optional): learning rate scheduler method. Default is single_step.
        stepsize (int or list, optional): step size to decay learning rate. When ``lr_scheduler``
            is "single_step", ``stepsize`` should be an integer. When ``lr_scheduler`` is
            "multi_step", ``stepsize`` is a list. Default is 1.
        gamma (float, optional): decay rate. Default is 0.1.
        max_epoch (int, optional): maximum epoch (for cosine annealing). Default is 1.

    Examples::
        >>> # Decay learning rate by every 20 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='single_step', stepsize=20
        >>> )
        >>> # Decay learning rate at 30, 50 and 55 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
        >>> )
    """
    if lr_scheduler not in AVAI_SCH:
        raise ValueError('Unsupported scheduler: {}. Must be one of {}'.format(lr_scheduler, AVAI_SCH))
    
    if lr_scheduler == 'single_step':
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]
        
        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'warmup_db':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For warmup_db lr_scheduler, stepsize must'
                'be a list, but got {}'.format(type(stepsize))
            )
        if not len(stepsize)==3:
            raise TypeError(
                'For warmup lr_scheduler, must have only 3 elements'
                'be a list, but got {}'.format(stepsize)
            )
        warmup_lambda = lambda epoch: warmup_db(epoch, stepsize)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[warmup_lambda]
        )

    elif lr_scheduler == 'warmup_sb':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For warmup_sb lr_scheduler, stepsize must'
                'be a list, but got {}'.format(type(stepsize))
            )
        if not len(stepsize)==3:
            raise TypeError(
                'For warmup lr_scheduler, must have only 3 elements'
                'be a list, but got {}'.format(stepsize)
            )
        warmup_lambda = lambda epoch: warmup_sb(epoch, stepsize)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[warmup_lambda]
        )

    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )

    return scheduler