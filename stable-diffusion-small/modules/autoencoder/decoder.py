import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_block import VAttentionBlock
from .residual_block import VResidualBlock


class UpScale(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpScale, self).__init__()

        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor
        )

        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upconv(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(
        self,
        base_channels=128,
        num_groups=32,
    ):
        super(Up, self).__init__()

        self.upsample = nn.Sequential(
            # f=8
            nn.Conv2d(
                base_channels // 32, base_channels // 32, kernel_size=1, padding=0
            ),
            nn.Conv2d(base_channels // 32, base_channels * 4, kernel_size=3, padding=1),
            VResidualBlock(base_channels * 4, base_channels * 4),
            VAttentionBlock(base_channels * 4),
            VResidualBlock(base_channels * 4, base_channels * 4),
            VResidualBlock(base_channels * 4, base_channels * 4),
            VResidualBlock(base_channels * 4, base_channels * 4),
            # f=4
            UpScale(base_channels * 4, base_channels * 4),
            VResidualBlock(base_channels * 4, base_channels * 4),
            VResidualBlock(base_channels * 4, base_channels * 4),
            VResidualBlock(base_channels * 4, base_channels * 4),
            # f=2
            UpScale(base_channels * 4, base_channels * 4),
            VResidualBlock(base_channels * 4, base_channels * 2),
            VResidualBlock(base_channels * 2, base_channels * 2),
            VResidualBlock(base_channels * 2, base_channels * 2),
            # f=1
            UpScale(base_channels * 2, base_channels * 2),
            VResidualBlock(base_channels * 2, base_channels),
            VResidualBlock(base_channels, base_channels),
            VResidualBlock(base_channels, base_channels),
            nn.GroupNorm(num_groups, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return x


class VDecoder(nn.Module):
    def __init__(
        self,
        base_channels=128,
        num_groups=32,
        z_scale_factor=0.18215,
    ):
        super(VDecoder, self).__init__()
        self.up = Up(base_channels=base_channels, num_groups=num_groups)
        self.z_scale_factor = z_scale_factor

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.up(z)
        x = x / self.z_scale_factor
        return x
