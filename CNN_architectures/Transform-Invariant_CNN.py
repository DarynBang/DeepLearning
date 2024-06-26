import torch
import torch.nn as nn
from torch import Tensor
import albumentations as A
from icecream import ic


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 **kwargs) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class RandomTransformations(nn.Module):
    def __init__(self,
                 height: int,
                 width: int,
                 test: bool = False) -> None:
        super(RandomTransformations, self).__init__()

        self.height = height
        self.width = width
        self.test = test

        self.random_transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.8),
            A.RandomScale(scale_limit=0.2, p=0.8),
            A.Rotate(limit=20, p=0.8),

            # Pad the image with zeros if it's smaller than the specified height and width
            A.PadIfNeeded(min_height=self.height, min_width=self.width, value=0, p=1.0),

            # Crop the image to the original size
            A.RandomCrop(height=self.height, width=self.width, p=1.0)

        ], additional_targets = {'image': 'image'})


    def forward(self, x) -> Tensor:
        B, C, H, W = x.size()
        transformed_imgs = []

        # Iterate over each image in the batch
        if self.test:
            return x
        else:
            for i in range(B):
                image = x[i].permute(1, 2, 0).detach().numpy()
                transformed = self.random_transform(image=image)['image']

                transformed = torch.tensor(transformed, device=x.device).permute(2, 0, 1).to(x.device)
                transformed_imgs.append(transformed.requires_grad_(x.requires_grad))

            # Stack the transformed images back into a batch tensor to return
            return torch.stack(transformed_imgs)

class TI_layer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 height: int,
                 width: int,
                 test: bool = False) -> None:
        super(TI_layer, self).__init__()
        self.test = test
        self.random_transformation = RandomTransformations(height, width, self.test)

        # padding = kernel_size // 2 to maintain the spatial dimensions of the input feature maps after the convolution operation
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x) -> Tensor:
        out = self.random_transformation(x)
        out = self.conv(out)

        return out


class Network(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 height: int,
                 width: int):
        super(Network, self).__init__()

        self.c1 = ConvBlock(in_channels, 16, 3, stride=1, padding=1)
        self.c2 = ConvBlock(16, 32, 3, stride=1, padding=1)

        self.ti_layer1 = TI_layer(32, 64, 3, height, width)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.ti_layer2 = TI_layer(64, 64, 3, height//2, width//2)

        self.c3 = ConvBlock(64, 128, 3, stride=1, padding=1)


        self.fc = nn.Linear(128 * 32 * 32, num_classes)

    def forward(self, x) -> Tensor:
        out = self.c1(x)
        out = self.c2(out)

        out = self.ti_layer1(out)

        out = self.maxpool(out)

        out = self.ti_layer2(out)

        out = self.c3(out)

        out = self.maxpool(out)

        out = torch.flatten(out, 1)

        return self.fc(out)


IMG_HEIGHT, IMG_WIDTH = 128, 128

model = Network(3, 8, IMG_HEIGHT, IMG_WIDTH)

x = torch.rand(1, 3, IMG_HEIGHT, IMG_WIDTH)

output = model(x)
print(output)
print(output.shape)
