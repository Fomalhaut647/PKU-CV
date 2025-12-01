import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out


class ResBlock(nn.Module):
    """residual block"""

    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        """
        in_channel: number of channels in the input image.
        out_channel: number of channels produced by the convolution.
        stride: stride of the convolution.
        """
        # 1. define double convolution
        # convolution
        # batch normalization
        # activate function
        # ......

        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.

        # Note: we are going to implement 'Basic residual block' by above steps, you can also implement 'Bottleneck Residual block'

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        return out


class ResNet(nn.Module):
    """residual network"""

    def __init__(self):
        super().__init__()

        # 1. define convolution layer to process raw RGB image
        # 2. define multiple residual blocks
        # 3. define full-connected layer to classify

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        return out


class ResNextBlock(nn.Module):
    """ResNext block"""

    def __init__(self, in_channel, out_channel, bottle_neck, group, stride):
        super().__init__()
        # in_channel: number of channels in the input image
        # out_channel: number of channels produced by the convolution
        # bottle_neck: int, bottleneck= out_channel / hidden_channel
        # group: number of blocked connections from input channels to output channels
        # stride: stride of the convolution.

        # 1. define convolution
        # 1x1 convolution
        # batch normalization
        # activate function
        # 3x3 convolution
        # ......
        # 1x1 convolution
        # ......

        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        return out


class ResNext(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define convolution layer to process raw RGB image
        # 2. define multiple residual blocks
        # 3. define full-connected layer to classify

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        return out
