"""
Custom loss functions for multi-label classification.

Reference: https://arxiv.org/abs/2009.14119
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Focal Loss for multi-label classification.

    Args:
        gamma_pos: Focus parameter for positive samples (usually 0).
        gamma_neg: Focus parameter for negative samples (e.g., 4).
                   Higher values = more aggressive down-weighting of easy negatives.
        clip: Clipping value for negative probabilities to prevent log(0).
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Focal weighting
        pt = xs_pos * targets + xs_neg * (1 - targets)
        one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        focal_weight = (1.0 - pt).pow(one_sided_gamma)

        # BCE terms
        loss = -(
            targets * torch.log(xs_pos.clamp(min=self.eps))
            + (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        )
        loss = loss * focal_weight
        return loss.mean()
