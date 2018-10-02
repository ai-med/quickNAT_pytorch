import torch
import torch.nn as nn


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in::

         Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507
    """

    def __int__(self, reduction_ratio=2):
        self.reduction_ratio = reduction_ratio
        super(ChannelSELayer, self).__init__()

    def forward(self, input: torch.Tensor):
        input_rank = len(input.shape)
        reduce_indices = list(range(input))[1:-1]
        squeeze_tensor = input.mean(dim=reduce_indices)

        # channel excitation
        num_channels = int(squeeze_tensor.shape[-1])
        reduction_ratio = self.reduction_ratio
        if num_channels % reduction_ratio != 0:
            raise ValueError("reduction ratio incompatible with number of input tensor channels")
        num_channels_reduced = num_channels / reduction_ratio
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()
        fc1 = nn.Linear(num_channels, num_channels_reduced, bias=False)
        fc2 = nn.Linear(num_channels_reduced, num_channels, bias=False)

        fc_out_1 = relu(fc1(squeeze_tensor))
        fc_out_2 = sigmoid(fc2(fc_out_1))

        with len(fc_out_2.shape) < input_rank:
            fc_out_2 = fc_out_2.unsqueeze(1)

        output_tensor = torch.mul(input, fc_out_2)
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially
    and exciting channel-wise described in::

        Roy et al., Concurrent Spatial and Channel Squeeze & Excitation
        in Fully Convolutional Networks, MICCAI 2018


    """

    def __init__(self):
        super(SpatialSELayer, self).__init__()

    def forward(self, input):
        # spatial squeeze
        num_channels = input.shape[1]
        conv = nn.Conv2d(num_channels, 1, 1)
        sigmoid = nn.Sigmoid()
        squeeze_tensor = sigmoid(conv(input))
        # spatial excitation
        output_tensor = torch.mul(input, squeeze_tensor)

        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel
    squeeze & excitation::

        Roy et al., Concurrent Spatial and Channel Squeeze & Excitation
        in Fully Convolutional Networks, arXiv:1803.02579

    """

    def __init__(self,
                 reduction_ratio=2):
        self.reduction_ratio = reduction_ratio
        super(ChannelSpatialSELayer, self).__init__()

    def forward(self, input):
        cSE = ChannelSELayer(self.reduction_ratio)
        sSE = SpatialSELayer()
        output_tensor = torch.max(cSE(input), sSE(input))
        return output_tensor
