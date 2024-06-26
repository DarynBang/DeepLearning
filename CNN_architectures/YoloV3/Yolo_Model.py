import torch
import torch.nn as nn
from torch import Tensor
import os

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 use_batch_norm: bool=True,
                 **kwargs) -> None:
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias= not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

        self.use_batch_norm = use_batch_norm


    def forward(self, x):
        f = self.conv(x)

        if self.use_batch_norm:
            f = self.bn(f)

        return self.activation(f)


class ResidualBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 use_residual: bool = True,
                 num_repeats: int = 1):
        super().__init__()

        resnet_layers = []

        for _ in range(num_repeats):
            resnet_layers += [
                nn.Sequential(
                    ConvBlock(channels, channels // 2, 3),
                    ConvBlock(channels // 2, channels, 3, padding=1)
                )
            ]

        self.layers = nn.ModuleList(resnet_layers)
        self.use_residual = use_residual
        self.num_repeats = num_repeats


    def forward(self, x):
        pass



print(os.getcwd())

