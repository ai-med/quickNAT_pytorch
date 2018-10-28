import torch
import numpy as np
from torch.nn.modules.loss import _Loss, _WeightedLoss
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

class DiceLoss(_WeightedLoss):
    def forward(self, output, target, weights=None, ignore_index=None):
        """
            output : NxCxHxW Variable
            target :  NxHxW LongTensor
            weights : C FloatTensor
            ignore_index : int index to ignore from loss
            """
        eps = 0.0001
        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)


class CrossEntropyLoss2d(_WeightedLoss):
    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


class CombinedLoss(_Loss):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()

    def forward(self, input, target, weight):
        # TODO: why?
        #target = target.type(torch.LongTensor).cuda()
        input_soft = F.softmax(input, dim = 1)
        y2 = torch.mean(self.dice_loss(input_soft, target))
        y1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        y = y1 + y2
        return y

