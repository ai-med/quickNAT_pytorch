import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F


class DiceCoeff(nn.Module):
    """Dice coeff for individual examples"""

    def __init__(self):
        super(DiceCoeff, self).__init__()

    def forward(self, input, target):
        inter = torch.dot(input, target) + 0.0001
        union = torch.sum(input ** 2) + torch.sum(target ** 2) + 0.0001

        t = 2 * inter.float() / union.float()
        return t


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = Variable(torch.FloatTensor(1).cuda().zero_())
    else:
        s = Variable(torch.FloatTensor(1).zero_())

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class DiceLoss(_Loss):
    def forward(self, input, target):
        return 1 - dice_coeff(input, target)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()

    def forward(self, input, target, weight):
        # TODO: why?
        # target_bin = target_bin.type(torch.FloatTensor).cuda()
        target = target.type(torch.LongTensor).cuda()
        #y = torch.mean(self.cross_entropy_loss.forward(input, target))
        y = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        return y
