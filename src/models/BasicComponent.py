"""Basic Component for convolution and deconvolution layer.
"""
import torch.nn as nn


class VGGMaxUnpool(nn.Module):
    """MaxPool layer based on Maxpooling Layers of VGG16.
    """
    def __init__(self):
        super(VGGMaxUnpool, self).__init__()
        self.maxupool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs, indices):
        return self.maxupool(inputs, indices=indices)


class VGGConvTranspose(nn.Module):
    """2D Transposed Convolutional layer + BN based on the conv2d layers
    of VGG16.
    """
    def __init__(self, in_channels, out_channels):
        super(VGGConvTranspose, self).__init__()
        self.convt = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.ac = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.ac(self.bn(self.convt(inputs)))


class Reshape(nn.Module):
    """Reshape Layer.
    """
    def __init__(self, args):
        super(Reshape, self).__init__()
        self.ex_shape = args

    def forward(self, inputs):
        shape = (inputs.size(0), )
        for index in self.ex_shape:
            shape += (index, )
        return inputs.view(*shape)


def VGGConv(in_channels, out_channels):
    """Returns a VGG16 Conv2D with BN and ReLU.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),\
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)


def VGGMaxPool():
    """Returns a VGG16 MaxPool.
    """
    return nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
