import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_block import VAttentionBlock
from .residual_block import VResidualBlock


class Conv2Pad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2Pad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # padding (left, right, top, bottom)
        return self.conv(F.pad(x, [0, 1, 0, 1]))


class Down(nn.Module):
    def __init__(
        self,
        base_channels=128,
        num_groups=32,
    ):
        super(Down, self).__init__()

        self.downsample = nn.Sequential(
            # f=1
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            VResidualBlock(base_channels, base_channels),
            VResidualBlock(base_channels, base_channels),
            # f=2
            Conv2Pad(base_channels, base_channels, kernel_size=3, stride=2, padding=0),
            VResidualBlock(base_channels, base_channels * 2),
            VResidualBlock(base_channels * 2, base_channels * 2),
            # f=4
            Conv2Pad(
                base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=0
            ),
            VResidualBlock(base_channels * 2, base_channels * 4),
            VResidualBlock(base_channels * 4, base_channels * 4),
            # f=8
            Conv2Pad(
                base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=0
            ),
            VResidualBlock(base_channels * 4, base_channels * 4),
            VResidualBlock(base_channels * 4, base_channels * 4),
            VResidualBlock(base_channels * 4, base_channels * 4),
            VAttentionBlock(base_channels * 4),
            VResidualBlock(base_channels * 4, base_channels * 4),
            nn.GroupNorm(num_groups, base_channels * 4),
            nn.SiLU(),
            # f=8
            nn.Conv2d(base_channels * 4, base_channels // 32, kernel_size=3, padding=1),
            # f=8
            nn.Conv2d(
                base_channels // 32, base_channels // 32, kernel_size=3, padding=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        return x


class VEncoder(nn.Module):
    def __init__(
        self,
        base_channels=128,
        num_groups=32,
        z_scale_factor=0.18215,
    ):
        super(VEncoder, self).__init__()
        self.down = Down(base_channels=base_channels, num_groups=num_groups)
        self.z_scale_factor = z_scale_factor

    def forward(self, x: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        x = self.down(x)
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -20, 20)
        std = log_var.exp().sqrt()

        if noise is None:
            noise = torch.randn_like(mean)

        z = mean + std * noise
        z = z * self.z_scale_factor

        return z, mean, log_var
