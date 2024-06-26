import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 **kwargs) -> None:
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn(self.conv(x))

        return out


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int) -> None:
        super(ResBlock, self).__init__()

        res_channels = in_channels // 4

        self.conv1 = ConvBlock(in_channels, res_channels, 1, stride=1, padding=0)
        self.conv2 = ConvBlock(res_channels, res_channels, 3, stride=1, padding=1)
        self.conv3 = ConvBlock(res_channels, in_channels, 1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)

        out = self.relu(torch.add(out, x))

        return out


# Implement UP part of Unet model
class DOWN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super(DOWN, self).__init__()

        self.down_conv_layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels//2, kernel_size=3, padding=1),
            ResBlock(out_channels // 2),
            ConvBlock(out_channels // 2, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_conv_layers(x)


class UP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super(UP, self).__init__()

        ## self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2), -> Uses Transpose2d to upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)


        self.Conv_layers = nn.Sequential(
            ConvBlock(in_channels, out_channels // 2, 3, padding=1),
            ResBlock(out_channels // 2),
            ConvBlock(out_channels // 2, out_channels, 3, padding=1)
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # x1 is input feature map from previous layer
        # x2 is the feature map from the contracting path (downsampling path)

        x1 = self.upsample(x1)

        # The dimensions of input are [0: batch_size, 1: n_channels, 2: height, 3: width]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]


        # Apply symmetric padding to both width and height
        ## pad = [pad_left, pad_right, pad_top, pad_bottom] -> [1, 1, 1, 1] means that 1 padding will be added to each side of x1.
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                             diffY // 2, diffY - diffY // 2])

        out = torch.cat([x2, x1], 1)
        return self.Conv_layers(out)


class OutConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)



class Unet(nn.Module):
    def __init__(self,
                 n_channels: int,
                 num_classes: int) -> None:
        super(Unet, self).__init__()
        self.doubleconv = nn.Sequential(
            ConvBlock(n_channels, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64, kernel_size=3, padding=1)
        )

        self.down1 = DOWN(64, 128)
        self.down2 = DOWN(128, 256)
        self.down3 = DOWN(256, 512)

        self.factor = 2

        # Note: divide by factor of 2 because the feature maps will be concatenated later on so their channels will be multiplied by 2
        self.down4 = DOWN(512, 1024 // self.factor)

        self.up1 = UP(1024, 512 // self.factor)
        self.up2 = UP(512, 256 // self.factor)
        self.up3 = UP(256, 128 // self.factor)
        self.up4 = UP(128, 64)

        self.output = OutConv(64, num_classes)


    def forward(self, x: Tensor) -> Tensor:
        x1 = self.doubleconv(x)

        ## Note: down functions goes in order of Downsample -> Conv -> ResBlock -> Conv
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.down4(x4)   # x5 is the bottleneck layer

        ## Note: up functions goes in order of Upsample -> Conv -> ResBlock -> Conv -> concatenate(x1, x2)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.output(x)

        return logits


if __name__ == '__main__':
    x = torch.rand(1, 3, 128, 128)

    model = Unet(3, 4)

    output = model(x)
    print(output.shape)


