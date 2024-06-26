import torch.nn as nn
import torch.nn.functional as f
from icecream import ic
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
import torch

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,
                x: Tensor) -> Tensor:
        return self.bn(self.conv(x))


class SE_Block(nn.Module):
    def __init__(self,
                 n_channels: int,
                 ratio: int) -> None:
        super(SE_Block, self).__init__()
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features = n_channels, out_features = (n_channels // ratio), bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(in_features = (n_channels // ratio), out_features = n_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                x: Tensor) -> Tensor:
        n_batches, n_channels, _, _ = x.size()
        # Squeeze Operation
        out = self.GlobalAvgPool(x).view(n_batches, n_channels)

        # Excitation operation
        out = self.relu(self.fc1(out))

        out = self.sigmoid(self.fc2(out))

        ## Reshape output to match the input dimensions
        out = out.view(n_batches, n_channels, 1, 1)

        return x * out.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super(ResBlock, self).__init__()
        res_channels = in_channels // 4
        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0)
        self.c2 = ConvBlock(res_channels, res_channels, 3, 1, 1)
        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)
        self.se_block = SE_Block(out_channels, 8)
        self.relu = nn.ReLU()

    def forward(self,
                x: Tensor) -> Tensor:
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)
        f = self.se_block(f)
        f = self.relu(torch.add(f, x))

        return f

## Model combines Residual blocks, batch normalization, dropout regularization
class Network(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int) -> None:
        super(Network, self).__init__()
        self.c1 = ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.res_block1 = ResBlock(in_channels=64, out_channels=64)

        self.c2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.res_block2 = ResBlock(in_channels=128, out_channels=128)

        self.c3 = ConvBlock(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.res_block3 = ResBlock(in_channels=192, out_channels=192)

        self.c4 = ConvBlock(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.res_block4 = ResBlock(in_channels=256, out_channels=256)

        self.c5 = ConvBlock(in_channels=256, out_channels=320, kernel_size=3, stride=1, padding=1)
        self.res_block5 = ResBlock(in_channels=320, out_channels=320)

        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=320 * 7 * 7, out_features=num_classes)

    def forward(self,
                x: Tensor) -> [Tensor, Tensor]:
        output = self.relu(self.c1(x))
        output = self.res_block1(self.dropout(output))

        output = self.relu(self.c2(output))
        output = self.res_block2(self.dropout(output))
        output = self.pool(output)

        output = self.relu(self.c3(output))
        output = self.res_block3(self.dropout(output))
        output = self.pool(output)

        output = self.relu(self.c4(output))
        output = self.res_block4(self.dropout(output))
        output = self.pool(output)

        output = self.relu(self.c5(output))
        output = self.res_block5(self.dropout(output))
        output = self.pool(output)

        output = torch.flatten(output, 1)
        logits = self.fc1(output)

        probas = f.softmax(logits, dim=1)
        return logits, probas



if __name__ == "__main__":
    x = torch.rand(1, 3, 128, 128)
    model = Network(in_channels=3, num_classes=10)
    logit, prob = model(x)
    ic(logit.shape, prob.shape)
    ic(logit)
    ic(prob)
