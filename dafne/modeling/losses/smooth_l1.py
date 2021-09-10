import math
from dafne.modeling.losses.utils import weighted_loss

import torch
from fvcore.nn import smooth_l1_loss
from dafne.utils.sort_corners import sort_quadrilateral
from torch import nn


@weighted_loss
def smooth_l1(pred, target, beta):
    return smooth_l1_loss(pred, target, beta, "none")

class SmoothL1Loss(nn.Module):
    """Weighted version of smooth_l1_loss."""

    def __init__(self, beta=1.0 / 9, reduction="sum", logspace=True):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.logspace = logspace

    def forward(self, input, target, weight=None):
        # Get original smooth_l1_loss without reduction
        loss = smooth_l1_loss(input, target, self.beta, reduction="None")

        if self.logspace:
            loss = loss.log1p()

        # Scale by weights if given
        if weight is not None and weight.sum() > 0:
            loss *= weight[:, None]

        # Perform reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class ModulatedEightPointLoss(nn.Module):
    def __init__(self, beta=1.0 / 9, reduction="sum", logspace=True):
        super(ModulatedEightPointLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.logspace = logspace

    def _smooth_l1_loss(
        self, input: torch.Tensor, target: torch.Tensor, reduction: str = "none"
    ) -> torch.Tensor:
        """See https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py"""
        if self.beta < 1e-5:
            # if self.beta == 0, then torch.where will result in nan gradients when
            # the chain rule is applied due to pytorch implementation details
            # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
            # zeros, rather than "no gradient"). To avoid this issue, we define
            # small values of self.beta to be exactly l1 loss.
            loss = torch.abs(input - target)
            # loss = l1_abs
        else:
            n = torch.abs(input - target)
            # n = l1_abs
            cond = n < self.beta
            loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)

        return loss

    def forward(self, input, target, weight=None):

        # Convert xywha to corner representation and sort them canonically

        num_pos = input.shape[0]

        # Loss without shift
        loss_0 = self._smooth_l1_loss(input, target)

        input = input.view(num_pos, 4, 2)

        # Clockwise shift
        input_tmp = input[:, [1, 2, 3, 0]].view(num_pos, -1)
        loss_1 = self._smooth_l1_loss(input_tmp, target)

        # Clockwise shift
        input_tmp = input[:, [3, 0, 1, 2]].view(num_pos, -1)
        loss_2 = self._smooth_l1_loss(input_tmp, target)

        if self.logspace:
            loss_0 = loss_0.log1p()
            loss_1 = loss_1.log1p()
            loss_2 = loss_2.log1p()

        if self.reduction == "mean":
            loss_0 = loss_0.mean(1)
            loss_1 = loss_1.mean(1)
            loss_2 = loss_2.mean(1)
        elif self.reduction == "sum":
            loss_0 = loss_0.sum(1)
            loss_1 = loss_1.sum(1)
            loss_2 = loss_2.sum(1)

        losses = torch.min(torch.stack((loss_0, loss_1, loss_2), dim=-1), dim=-1).values

        # Scale by weights if given
        if weight is not None and weight.sum() > 0:
            losses *= weight

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        return losses


class ModulatedSmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0 / 9, reduction="sum", logspace=True):
        super(ModulatedSmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.beta = beta
        self.logspace = logspace

    def _smooth_l1_loss(
        self, l1_abs: torch.Tensor, reduction: str = "none"
    ) -> torch.Tensor:
        """See https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py"""
        if self.beta < 1e-5:
            # if self.beta == 0, then torch.where will result in nan gradients when
            # the chain rule is applied due to pytorch implementation details
            # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
            # zeros, rather than "no gradient"). To avoid this issue, we define
            # small values of self.beta to be exactly l1 loss.
            loss = torch.log1p(l1_abs)
            # loss = l1_abs
        else:
            n = torch.log1p(l1_abs)
            # n = l1_abs
            cond = n < self.beta
            loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        return loss

    def forward(self, input, target, weight=None):
        # Convert degrees to radians
        input[:, 4] = input[:, 4] / 180.0 * math.pi
        target[:, 4] = target[:, 4] / 180.0 * math.pi

        # Loss without changes
        # |x1 - x2| + |y1 - y2| + |w1 - w2| + |h1 - h2| + |theta1 - theta2|
        l1_abs = torch.abs(input - target)
        loss_0 = self._smooth_l1_loss(l1_abs)

        # Loss with h/w swap and 90 - angle
        # |x1 - x2| + |y1 - y2| + |w1 - h2| + |h1 - w2| + |pi/2 - |theta1 - theta2||
        input_tmp = input[:, [0, 1, 3, 2, 4]]
        l1_abs = torch.abs(input_tmp - target)
        l1_abs[:, 4] = torch.abs(math.pi / 2.0 - l1_abs[:, 4])
        loss_1 = self._smooth_l1_loss(l1_abs)

        if self.logspace:
            loss_0 = loss_0.log1p()
            loss_1 = loss_1.log1p()
            loss_2 = loss_2.log1p()

        # Reduce over all elements (x, y, w, h, theta) in a single box
        if self.reduction == "mean":
            loss_0 = loss_0.mean(1)
            loss_1 = loss_1.mean(1)
        elif self.reduction == "sum":
            loss_0 = loss_0.sum(1)
            loss_1 = loss_1.sum(1)

        losses = torch.min(torch.stack((loss_0, loss_1), dim=-1), dim=-1).values

        # Scale by weights if given
        if weight is not None and weight.sum() > 0:
            losses *= weight

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        return losses
