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


    def forward(self, x) -> Tensor:
        f = self.conv(x)

        if self.use_batch_norm:
            f = self.bn(f)

        return self.activation(f)


class SEBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 reduction_ratio: int) -> None:
        super().__init__()
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=channels, out_features=channels//reduction_ratio, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(in_features=channels//reduction_ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        f = self.GlobalAvgPool(x).view(B, C)

        f = self.relu(self.fc1(f))
        f = self.sigmoid(self.fc2(f))

        output = f.view(B, C, 1, 1)

        return x * output


class ResidualBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 use_residual: bool = True,
                 use_SE: bool = False,
                 num_repeats: int = 1,) -> None:
        super().__init__()
        self.use_SE = use_SE

        resnet_layers = []

        for _ in range(num_repeats):
            if not use_SE:
                resnet_layers += [
                    nn.Sequential(
                        ConvBlock(channels, channels // 2, 3),
                        ConvBlock(channels // 2, channels, 3, padding=1)
                    )
                ]

            else:
                resnet_layers += [
                    nn.Sequential(
                        ConvBlock(channels, channels // 2, 3),
                        SEBlock(channels // 2, 4),
                        ConvBlock(channels // 2, channels, 3, padding=1)
                    )
                ]


        self.layers = nn.ModuleList(resnet_layers)
        self.use_residual = use_residual
        self.num_repeats = num_repeats


    def forward(self, x) -> Tensor:
        for layer in self.layers:
            residual = x
            x = layer(x)
            if self.use_residual:
                x = x + residual

        return x

class ScalePrediction(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int) -> None:
        super().__init__()
        self.pred = nn.Sequential(
            ConvBlock(in_channels, in_channels*2, 3, padding=1),
            nn.Conv2d(2*in_channels, in_channels, (num_classes + 5) * 3, kernel_size=1)
        )

        self.num_classes = num_classes

    def forward(self , x) -> Tensor:
        B, C, H, W = x.size()
        f = self.pred(x)
        f = f.view(B, 3, self.num_classes + 5, H, W)

        output = f.permute(0, 1, 3, 4, 2)

        return output
