import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def elu1(x, inplace=True, eps=0.0):
    return F.elu(x, inplace=inplace) + 1.0 + eps

def adaptive_elu(x, xshift, yshift):
    return F.elu(x - xshift, inplace=True) + yshift

class Elu1(nn.Module):
    """
    Elu activation function shifted by 1 to ensure that the
    output stays positive. That is:
    Elu1(x) = Elu(x) + 1
    """

    def forward(self, x, inplace=True, eps=0):
        return elu1(x, inplace, eps)


class AdaptiveELU(nn.Module):
    """
    ELU shifted by user specified values. This helps to ensure the output to stay positive.
    """

    def __init__(self, xshift, yshift, **kwargs):
        super(AdaptiveELU, self).__init__(**kwargs)

        self.xshift = xshift
        self.yshift = yshift

    def forward(self, x):
        return adaptive_elu(x, self.xshift, self.yshift)
