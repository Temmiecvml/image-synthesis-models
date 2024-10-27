import torch
import torch.nn as nn
import torch.nn.functional as F


class VResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_groups=32,
    ):
        super(VResidualBlock, self).__init__()
        self.group_norm1 = nn.GroupNorm(num_groups, in_channels)
        self.group_norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.res = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x):
        h = F.silu(self.group_norm1(x))
        h = self.conv1(h)
        h = F.silu(self.group_norm2(h))
        h = self.conv2(h)
        h += self.res(x)

        return h
