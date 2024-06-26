import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class ConvBlock(nn.Module):
    def __init___(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        f = self.relu(self.bn(self.conv(x)))

        return f


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out1x1, reduced3x3, out3x3, reduced5x5, out5x5, out1x1pool):
        super(InceptionBlock, self).__init__()
        ## 1x1 branch
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=out1x1, kernel_size=1)

        ## 1x1 followed by 3x3 branch
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=reduced3x3, kernel_size=1),
            ConvBlock(in_channels=reduced3x3, out_channels=out3x3, kernel_size=3, stride=1, padding=1)
        )

        ## 1x1 followed by 5x5 branch
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=reduced5x5, kernel_size=1),
            ConvBlock(in_channels=reduced5x5, out_channels=out5x5, stride=1, padding=2)
        )

        ## Max pooling followed by 1x1 branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=out1x1pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class GoogLeNet(nn.Module):
    def __init_(self, in_channels, num_classes):
        super(GoogLeNet, self).__init__()

        self.conv1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=7, stride=2, padding=3)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)

        # In this order: in_channels, out1x1, reduced3x3, out3x3, reduced5x5, out5x5, out1x1pool

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        # self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.25)

        ## Calculate in_features tomorrow
        self.fc1 = nn.Linear(..., num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        ## Inception 3a
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        ## Inception 4a
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # x = self.inception5a(x)
        # x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        return x

if __name__ == "__main__":
    BATCH_SIZE = 5
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    model = GoogLeNet(aux_logits=True, num_classes=1000)
    print(model(x)[2].shape)
    assert model(x)[2].shape == torch.Size([BATCH_SIZE, 1000])




