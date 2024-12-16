import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, scale_factor=4):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])
        self.upsampling = nn.Sequential(
            *[UpsampleBlock(64) for _ in range(int(np.log2(scale_factor)))]
        )
        self.final = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        x = self.upsampling(x)
        x = self.final(x)
        return torch.tanh(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.blocks = nn.Sequential(
            discriminator_block(3, 64, 2),
            discriminator_block(64, 128, 2),
            discriminator_block(128, 256, 2),
            discriminator_block(256, 512, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)
