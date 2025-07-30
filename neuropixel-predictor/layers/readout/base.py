import warnings
from typing import Any, Literal, Mapping, Optional

import torch
from torch import nn as nn
from torch.nn.modules import Module
from torch.nn.parameter import Parameter

Reduction = Literal["sum", "mean", None]


class ConfigurationError(Exception):
    pass


# ------------------ Base Classes -------------------------


class Readout(Module):
    """
    Base readout class for all individual readouts.
    The MultiReadout will expect its readouts to inherit from this base class.
    """

    features: Parameter
    bias: Parameter

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    def regularizer(self, reduction: Reduction = "sum", average: Optional[bool] = None) -> torch.Tensor:
        raise NotImplementedError("regularizer is not implemented for ", self.__class__.__name__)

    def apply_reduction(
        self, x: torch.Tensor, reduction: Reduction = "mean", average: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Applies a reduction on the output of the regularizer.
        Args:
            x: output of the regularizer
            reduction: method of reduction for the regularizer. Currently possible are ['mean', 'sum', None].
            average: Deprecated. Whether to average the output of the regularizer.
                            If not None, it is transformed into the corresponding value of 'reduction' (see method 'resolve_reduction_method').

        Returns: reduced value of the regularizer
        """
        reduction = self.resolve_reduction_method(reduction=reduction, average=average)

        if reduction == "mean":
            return x.mean()
        elif reduction == "sum":
            return x.sum()
        elif reduction is None:
            return x
        else:
            raise ValueError(
                f"Reduction method '{reduction}' is not recognized. Valid values are ['mean', 'sum', None]"
            )

    def resolve_reduction_method(self, reduction: Reduction = "mean", average: Optional[bool] = None) -> Reduction:
        """
        Helper method which transforms the old and deprecated argument 'average' in the regularizer into
        the new argument 'reduction' (if average is not None). This is done in order to agree with the terminology in pytorch).
        """
        if average is not None:
            warnings.warn("Use of 'average' is deprecated. Please consider using `reduction` instead")
            reduction = "mean" if average else "sum"
        return reduction

    def resolve_deprecated_gamma_readout(
        self, feature_reg_weight: Optional[float], gamma_readout: Optional[float], default: float = 1.0
    ) -> float:
        if gamma_readout is not None:
            warnings.warn(
                "Use of 'gamma_readout' is deprecated. Use 'feature_reg_weight' instead. If 'feature_reg_weight' is defined, 'gamma_readout' is ignored"
            )

        if feature_reg_weight is None:
            if gamma_readout is not None:
                feature_reg_weight = gamma_readout
            else:
                feature_reg_weight = default
        return feature_reg_weight

    def initialize_bias(self, mean_activity: Optional[torch.Tensor] = None) -> None:
        """
        Initialize the biases in readout.
        Args:
            mean_activity: Tensor containing the mean activity of neurons.

        Returns:

        """
        if mean_activity is None:
            warnings.warn("Readout is NOT initialized with mean activity but with 0!")
            self.bias.data.fill_(0)
        else:
            self.bias.data = mean_activity

    def __repr__(self) -> str:
        return super().__repr__() + " [{}]\n".format(self.__class__.__name__)  # type: ignore[no-untyped-call,no-any-return]

