import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def laplace():
    """
    Returns a 3x3 laplace filter.

    """
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)[None, None, ...]

def laplace5x5():
    """
    Returns a 5x5 LaplacianOfGaussians (LoG) filter.

    """
    return np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0],]).astype(
        np.float32
    )[None, None, ...]

def laplace1d():
    return np.array([-1, 4, -1]).astype(np.float32)[None, None, ...]

def laplace3d():
    l = np.zeros((3, 3, 3))
    l[1, 1, 1] = -6.0
    l[1, 1, 2] = 1.0
    l[1, 1, 0] = 1.0
    l[1, 0, 1] = 1.0
    l[1, 2, 1] = 1.0
    l[0, 1, 1] = 1.0
    l[2, 1, 1] = 1.0
    return l.astype(np.float32)[None, None, ...]

class Laplace(nn.Module):
    """
    Laplace filter for a stack of data. Utilized as the input weight regularizer.
    """

    def __init__(self, padding=None, filter_size=3):
        """
        Laplace filter for a stack of data.

        Args:
            padding (int): Controls the amount of zero-padding for the convolution operation
                            Default is half of the kernel size (recommended)

        Attributes:
            filter (2D Numpy array): 3x3 Laplace filter.
            padding_size (int): Number of zeros added to each side of the input image
                before convolution operation.
        """
        super().__init__()
        if filter_size == 3:
            kernel = laplace()
        elif filter_size == 5:
            kernel = laplace5x5()
        elif filter_size == 7:
            kernel = laplace7x7()

        self.register_buffer("filter", torch.from_numpy(kernel))
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding

    def forward(self, x):
        return F.conv2d(x, self.filter, bias=None, padding=self.padding_size)

class Laplace1d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        filter = laplace1d()
        self.register_buffer("filter", torch.from_numpy(filter))
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding

    def forward(self, x):
        return F.conv1d(x, self.filter, bias=None, padding=self.padding_size)


class LaplaceL2(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer. Unnormalized, not recommended to use.
        Use LaplaceL2norm instead.

        Args:
            padding (int): Controls the amount of zero-padding for the convolution operation.

        Attributes:
            laplace (Laplace): Laplace convolution object. The output is the result of
                convolving an input image with laplace filter.

    """

    def __init__(self, padding=None, filter_size=3):
        super().__init__()
        self.laplace = Laplace(padding=padding, filter_size=filter_size)
        warnings.warn("LaplaceL2 Regularizer is deprecated. Use LaplaceL2norm instead.")

    def forward(self, x, avg=True):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.reshape(oc * ic, 1, k1, k2)).pow(2)) / 2


class LaplaceL2norm(nn.Module):
    """
    Normalized Laplace regularizer for a 2D convolutional layer.
        returns |laplace(filters)| / |filters|
    """

    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.reshape(oc * ic, 1, k1, k2)).pow(2)) / agg_fn(x.reshape(oc * ic, 1, k1, k2).pow(2))


class Laplace3d(nn.Module):
    """
    Laplace filter for a stack of data.
    """

    def __init__(self, padding=None):
        super().__init__()
        self.register_buffer("filter", torch.from_numpy(laplace3d()))

    def forward(self, x):
        return F.conv3d(x, self.filter, bias=None)


class LaplaceL2norm(nn.Module):
    """
    Normalized Laplace regularizer for a 2D convolutional layer.
        returns |laplace(filters)| / |filters|
    """

    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.reshape(oc * ic, 1, k1, k2)).pow(2)) / agg_fn(x.reshape(oc * ic, 1, k1, k2).pow(2))


class DepthLaplaceL21d(nn.Module):
    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace1d(padding=padding)

    def forward(self, x, avg=False):
        oc, ic, t = x.size()
        if avg:
            return torch.mean(self.laplace(x.reshape(oc * ic, 1, t)).pow(2)) / torch.mean(
                x.reshape(oc * ic, 1, t).pow(2)
            )
        else:
            return torch.sum(self.laplace(x.reshape(oc * ic, 1, t)).pow(2)) / torch.sum(x.reshape(oc * ic, 1, t).pow(2))
