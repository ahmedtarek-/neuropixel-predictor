import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Base class for all encoders
    """

    def regularizer(self, data_key=None, reduction="sum", average=None, detach_core=False):
        reg = self.core.regularizer().detach() if detach_core else self.core.regularizer()
        reg = reg + self.readout.regularizer(data_key=data_key, reduction=reduction, average=average)
        return reg

    def predict_mean(self, x, *args, data_key=None, **kwargs):
        raise NotImplementedError()

    def predict_variance(self, x, *args, data_key=None, **kwargs):
        raise NotImplementedError()


class GeneralizedEncoderBase(Encoder):
    def __init__(
        self, core, readout,
        nonlinearity_type_list=[],
        elu=False
    ):
        """
        An Encoder that wraps the core and readout into one model. Can predict any distribution.
        Args:
            core (nn.Module): Core model. Refer to neuralpredictors.layers.cores
            readout (nn.ModuleDict): MultiReadout model. Refer to neuralpredictors.layers.readouts
            nonlinearity_type_list (list of classes/functions): Non-linearity type to use.
            elu (bool): Use ELU and offset by 1 if True.
        """
        super().__init__()
        self.core = core
        self.readout = readout
        self.nonlinearity_type_list = nonlinearity_type_list
        self.elu = elu


    def forward(
        self,
        x,
        data_key=None,
        behavior=None,
        trial_idx=None,
        detach_core=False,
        **kwargs
    ):
        # get readout outputs
        x = self.core(x)
        if detach_core:
            x = x.detach()

        if "sample" in kwargs:
            x = self.readout(x, data_key=data_key, sample=kwargs["sample"])
        else:
            x = self.readout(x, data_key=data_key)

        # keep batch dimension if only one image was passed
        params = []
        for param in x:
            params.append(param[None, ...] if len(param.shape) == 1 else param)
        x = torch.stack(params)

        if self.elu:
            x = F.elu(x) + 1 # Because ELU returns -1 for all -ve values
            # x = F.elu(x)

        # assert len(self.nonlinearity_type_list) == len(x) == len(self.nonlinearity_config_list), (
        #     "Number of non-linearities ({}), number of readout outputs ({}) and, if available, number of non-linearity configs must match. "
        #     "If you do not wish to restrict a certain readout output with a non-linearity, assign the activation 'Identity' to it."
        # )
        # output = []
        # for i, (nonlinearity, out) in enumerate(zip(self.nonlinearity_type_list, x)):
        #     output.append(nonlinearity(out, **self.nonlinearity_config_list[i]))

        return x
