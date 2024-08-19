import torch
import torch.nn as nn
from torch import Tensor


NUM_CLASSES = 120
# Pretrained model (Test_2)
class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 use_bn: bool = True,
                 use_activation: bool = True,
                 **kwargs) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not use_bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

        self.use_bn = use_bn
        self.use_activation = use_activation
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x) -> Tensor:
        f = self.conv(x)

        if self.use_bn:
            f = self.bn(f)

        if self.use_activation:
            f = self.activation(f)

        return f


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class SeBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 reduction_ratio: int = 4) -> None:
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x) -> Tensor:
        B, C, H, W = x.shape

        y = self.squeeze(x).view(B, C)
        y = self.excitation(y).view(B, C, 1, 1)

        return torch.mul(x, y.expand_as(x))



class ResBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 ratio: int=4,
                 use_SE: bool = True):
        super().__init__()
        res_channels = channels // 4

        self.layers = nn.Sequential(
            ConvBlock(channels, res_channels, 1, use_activation=False, stride=1, padding=0),
            nn.ReLU(inplace=True),
            ConvBlock(res_channels, channels, 3, use_activation=False, stride=1, padding=1)
        )

        self.se_block = SeBlock(channels, ratio)
        self.use_SE = use_SE
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        f = self.layers(x)

        if self.use_SE:
            f = self.se_block(f)

        f = self.activation(torch.add(f, x))
        return f


class Network(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()


        self.layers = nn.ModuleList([
            ConvBlock(in_channels, 16, 5, use_bn=False, stride=1, padding=1),
            nn.Dropout(0.25),
            ConvBlock(16, 64, 3, stride=1, padding=1),

            ResBlock(64),

            ConvBlock(64, 128, 3, stride=1, padding=1),
            ResBlock(128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvBlock(128, 192, 3, stride=1, padding=1),
            ResBlock(192),
            ResBlock(192, use_SE=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvBlock(192, 256, 3, stride=1, padding=1),
            ResBlock(256, 8),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvBlock(256, 320, 3, use_bn=False, stride=1, padding=1),
            nn.Dropout(0.25),
            ResBlock(320, 8),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvBlock(320, 396, 3, stride=1, padding=1),
            ResBlock(396, 8),
            ResBlock(396, 4),

            ConvBlock(396, 448, 3, stride=1, padding=1),
            ResBlock(448, 8),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
          ])

        self.fc = nn.Linear(448*6*6, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        output = torch.flatten(x, 1)
        logits = self.fc(output)

        return logits


class UpdatedNetwork(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()


        self.layers = nn.ModuleList([
            ConvBlock(in_channels, 16, 5, use_bn=False, stride=1, padding=1),
            nn.Dropout(0.25),
            ConvBlock(16, 64, 3, stride=1, padding=1),

            ResBlock(64),

            ConvBlock(64, 128, 3, stride=1, padding=1),
            ResBlock(128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvBlock(128, 192, 3, stride=1, padding=1),
            ResBlock(192),
            ResBlock(192, use_SE=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvBlock(192, 256, 3, stride=1, padding=1),
            ResBlock(256, 8),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvBlock(256, 320, 3, use_bn=False, stride=1, padding=1),
            nn.Dropout(0.25),
            ResBlock(320, 8),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvBlock(320, 396, 3, stride=1, padding=1),
            ResBlock(396, 8),
            ResBlock(396, 4),

            ConvBlock(396, 448, 3, stride=1, padding=1),
            ResBlock(448, 8),
            Identity(),

            ConvBlock(448, 512, 3, stride=1, padding=1),
            ResBlock(512, 8),
            ResBlock(512, 8),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ConvBlock(512, 680, 3, stride=1, padding=1),
            ResBlock(680, 8),
            ResBlock(680, 8),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
          ])

        self.fc = nn.Linear(680*3*3, NUM_CLASSES)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        output = torch.flatten(x, 1)
        logits = self.fc(output)

        return logits