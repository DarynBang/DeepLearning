import torch.nn as nn
import torch
from torch import Tensor
import numpy as np

# Objective: Incorporate Inception + ResNets + BatchNorm + Dropout + Nin


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 **kwargs) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.c1(x))


class ResNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super(ResNetBlock, self).__init__()
        res_channels = in_channels // 4
        self.c1 = ConvBlock(in_channels, res_channels, 1, stride=1, padding=0)
        self.c2 = ConvBlock(res_channels, res_channels, 3, stride=1, padding=1)
        self.c3 = ConvBlock(res_channels, out_channels, 1, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)
        f = self.relu(torch.add(f, x))

        return f


class Stem(nn.Module):
    def __init__(self,
                 in_channels: int) -> None:
        super(Stem, self).__init__()
        self.conv_layer1 = ConvBlock(in_channels, 32, 5, stride=2, padding=0)
        self.conv_layer2 = ConvBlock(32, 64, 3, stride=1, padding=0)

        self.res_layer3 = ResNetBlock(64, 64)
        self.maxpool_layer = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_layer5 = ConvBlock(64, 152, 1, stride=1, padding=0)
        self.res_layer6 = ResNetBlock(152, 152)
        self.conv_layer7 = ConvBlock(152, 256, 3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.res_layer3(out)

        out = self.maxpool_layer(out)

        out = self.conv_layer5(out)
        out = self.res_layer6(out)

        out = self.conv_layer7(out)

        return out


class InceptionResNetA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 scale_factor: float) -> None:
        super(InceptionResNetA, self).__init__()
        self.scale_factor = scale_factor

        self.branch0 = ConvBlock(in_channels, 32, 1, stride=1, padding=0)

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 32, 1, stride=1, padding=0),
            ConvBlock(32, 32, 3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 32, 1, stride=1, padding=0),
            ConvBlock(32, 64, 3, stride=1, padding=1),
            ConvBlock(64, 128, 3, stride=1, padding=1)
        )

        self.conv = ConvBlock(192, 256, 1, stride=1, padding=0)

        ## n_out_channels = 32 + 32 + 128 = 192

        self.relu = nn.ReLU()

    def forward(self,
                x: Tensor) -> Tensor:
        out_branch0 = self.branch0(x)
        out_branch1 = self.branch1(x)
        out_branch2 = self.branch2(x)
        out = torch.cat([out_branch0, out_branch1, out_branch2], dim=1)
        out = self.conv(out)

        out = torch.mul(out, self.scale_factor)

        out = torch.add(out, x)
        out = self.relu(out)

        return out



class ReductionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 k: int,
                 l: int,
                 m: int,
                 n: int) -> None:
        super(ReductionBlock, self).__init__()
        self.branch0 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, n, 1, stride=1, padding=0),
            ConvBlock(n, n, 3, stride=2, padding=0)
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, k, 1, stride=1, padding=0),
            ConvBlock(k, l, 3, stride=1, padding=1),
            ConvBlock(l, m, 3, stride=2, padding=0)
        )

    def forward(self, x):
        out_branch0 = self.branch0(x)
        out_branch1 = self.branch1(x)
        out_branch2 = self.branch2(x)

        out = torch.cat([out_branch0, out_branch1, out_branch2], dim=1)

        return out


class InceptionResNetB(nn.Module):
    def __init__(self,
                 in_channels: int,
                 scale_factor: float) -> None:
        super(InceptionResNetB, self).__init__()
        self.scale_factor = scale_factor
        self.branch0 = nn.Sequential(
            ConvBlock(in_channels, 128, 1, stride=1, padding=0)
        )

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 64, 1, stride=1, padding=0),
            ConvBlock(64, 128, (1, 7), stride=1, padding=(0, 3)),
            ConvBlock(128, 128, (7, 1), stride=1, padding=(3, 0))
        )

        self.conv = ConvBlock(256, 672, 1, stride=1, padding=0)
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([self.branch0(x), self.branch1(x)], dim=1)
        out = self.conv(out)

        out = torch.mul(out, self.scale_factor)
        out = torch.add(out, x)
        out = self.relu(out)

        return out


class Model(nn.Module):
    def __init__(self,
                 k: int = 96,
                 l: int = 144,
                 m: int = 192,
                 n: int = 224,
                 num_classes: int = 10) -> None:
        super(Model, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.main_layers = nn.Sequential(
            # 2 max pooling layers
            Stem(in_channels=3),

            InceptionResNetA(in_channels=256, scale_factor=0.18),
            self.dropout,
            InceptionResNetA(in_channels=256, scale_factor=0.18),
            self.dropout,
            InceptionResNetA(in_channels=256, scale_factor=0.18),

            ReductionBlock(256, k, l, m, n),

            InceptionResNetB(in_channels=672, scale_factor=0.12),
            InceptionResNetB(in_channels=672, scale_factor=0.12),
            self.dropout,

            InceptionResNetB(in_channels=672, scale_factor=0.12),
            InceptionResNetB(in_channels=672, scale_factor=0.12),
            self.dropout,

            InceptionResNetB(in_channels=672, scale_factor=0.12)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(in_features=672, out_features=num_classes)

        self._initialize_weights()


    def forward(self, x) -> Tensor:
        out = self.main_layers(x)
        out = self.global_avg_pool(out)

        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out


    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                stddev = float(module.stddev) if hasattr(module, "stddev") else 0.1
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=stddev, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


x = torch.rand(1, 3, 64, 64) / 255
model = Model()
print(model(x))
print(model(x).shape)

