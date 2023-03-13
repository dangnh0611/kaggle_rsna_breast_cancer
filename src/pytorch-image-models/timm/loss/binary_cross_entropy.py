""" Binary Cross Entropy w/ a few extras

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCrossEntropy(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """
    def __init__(
            self, smoothing=0.1, target_threshold: Optional[float] = None, weight: Optional[torch.Tensor] = None,
            reduction: str = 'mean', pos_weight: Optional[torch.Tensor] = None):
        super(BinaryCrossEntropy, self).__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing = smoothing
        self.target_threshold = target_threshold
        self.reduction = reduction
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
            num_classes = x.shape[-1]
            # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
            if num_classes > 1:
                off_value = self.smoothing / num_classes
                on_value = 1. - self.smoothing + off_value
                target = target.long().view(-1, 1)
                target = torch.full(
                    (target.size()[0], num_classes),
                    off_value,
                    device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
            elif num_classes == 1:
                off_value = self.smoothing / 2
                on_value = 1. - self.smoothing + off_value
                target = torch.where(target.bool(), on_value, off_value).view(-1, 1)
            else:
                raise AssertionError()

        if self.target_threshold is not None:
            # Make target 0, or 1 if threshold set
            target = target.gt(self.target_threshold).to(dtype=target.dtype)

        return F.binary_cross_entropy_with_logits(
            x, target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction)



class BinaryCrossEntropyPosSmoothOnly(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """
    def __init__(
            self, smoothing=0.1, target_threshold: Optional[float] = None, weight: Optional[torch.Tensor] = None,
            reduction: str = 'mean', pos_weight: Optional[torch.Tensor] = None):
        super(BinaryCrossEntropyPosSmoothOnly, self).__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing = smoothing
        self.target_threshold = target_threshold
        self.reduction = reduction
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
            num_classes = x.shape[-1]
            # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
            assert num_classes <= 2
            if num_classes > 1:
                on_value = 1. - self.smoothing
                target = target.float().view(-1, 1).repeat(1, 2)
                target[:, 1] = target[:, 1] * on_value
                target[:, 0] = 1.0 - target[:, 1]
            elif num_classes == 1:
                off_value = 0
                on_value = 1. - self.smoothing
                target = torch.where(target.bool(), on_value, off_value).view(-1, 1)
            else:
                raise AssertionError()

        if self.target_threshold is not None:
            # Make target 0, or 1 if threshold set
            target = target.gt(self.target_threshold).to(dtype=target.dtype)

        return F.binary_cross_entropy_with_logits(
            x, target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction)

