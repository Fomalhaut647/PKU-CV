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

    def __init__(self, in_channel, out_channel, stride=1):
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

        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.

        if in_channel != out_channel or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H * W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """residual network"""

    def __init__(self):
        super().__init__()

        # 1. define convolution layer to process raw RGB image
        # 2. define multiple residual blocks
        # 3. define full-connected layer to classify

        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
        )

        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification

        x = self.relu(self.bn(self.conv(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out = self.fc(self.flatten(self.avgpool(x)))

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

        hidden_channel = out_channel // bottle_neck

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channel)

        self.conv2 = nn.Conv2d(
            hidden_channel,
            hidden_channel,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=group,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden_channel)

        self.conv3 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        if in_channel != out_channel or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channel, out_channel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNext(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define convolution layer to process raw RGB image
        # 2. define multiple residual blocks
        # 3. define full-connected layer to classify

        self.bottle_neck = 4
        self.group = 4

        self.conv = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(
            ResNextBlock(
                128, 256, bottle_neck=self.bottle_neck, group=self.group, stride=2
            ),
            ResNextBlock(
                256, 256, bottle_neck=self.bottle_neck, group=self.group, stride=1
            ),
            ResNextBlock(
                256, 256, bottle_neck=self.bottle_neck, group=self.group, stride=1
            ),
        )

        self.layer2 = nn.Sequential(
            ResNextBlock(
                256, 512, bottle_neck=self.bottle_neck, group=self.group, stride=2
            ),
            ResNextBlock(
                512, 512, bottle_neck=self.bottle_neck, group=self.group, stride=1
            ),
            ResNextBlock(
                512, 512, bottle_neck=self.bottle_neck, group=self.group, stride=1
            ),
        )

        self.layer3 = nn.Sequential(
            ResNextBlock(
                512, 1024, bottle_neck=self.bottle_neck, group=self.group, stride=2
            ),
            ResNextBlock(
                1024, 1024, bottle_neck=self.bottle_neck, group=self.group, stride=1
            ),
            ResNextBlock(
                1024, 1024, bottle_neck=self.bottle_neck, group=self.group, stride=1
            ),
        )

        self.layer4 = nn.Sequential(
            ResNextBlock(
                1024, 2048, bottle_neck=self.bottle_neck, group=self.group, stride=2
            ),
            ResNextBlock(
                2048, 2048, bottle_neck=self.bottle_neck, group=self.group, stride=1
            ),
            ResNextBlock(
                2048, 2048, bottle_neck=self.bottle_neck, group=self.group, stride=1
            ),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, 10)

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H * W]
        # extract features
        # classification

        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(self.flatten(self.avgpool(x)))
        return x
