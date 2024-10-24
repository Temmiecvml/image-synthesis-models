from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.autoencoder.decoder import UpScale


def reshape_for_attention(
    x: torch.Tensor, image_shape: Optional[Tuple[int, int]] = None
):
    if not image_shape:
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
    else:
        b, s, c = x.shape
        h, w = image_shape
        assert s == h * w, "Shape mismatch"
        x = x.permute(0, 2, 1).view(b, c, h, w)

    return x


class TimeEmbedding(nn.Module):
    def __init__(self, dims: int = 320):
        super().__init__()
        self.linear_1 - nn.Linear(dims, 4 * dims)
        self.linear_2 = nn.Linear(4 * dims, 4 * dims)

    def forward(self, t):
        x = self.linear_1(t)
        x = F.silu(x)
        x = self.linear_2(x)

        return x


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, t: torch.Tensor):

        for layer in self:
            if isinstance(layer, UNetCrossAttentionBlock):
                x = layer(x, context)

            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, t)

            else:
                x = layer(x)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self, num_heads: int, dims: int, context: int = 768, in_proj_bias: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dims = dims
        self.context = context
        self.in_proj_bias = in_proj_bias
        self.attention = nn.MultiheadAttention(
            dims, num_heads, batch_first=True, bias=False
        )
        self.linear_context = nn.Linear(context, dims, bias=False)
        self.linear_out = nn.Linear(dims, dims, bias=False)

    def forward(self, x, context):
        pass


class UNetCrossAttentionBlock(nn.Module):
    def __init__(self, num_heads: int, dims: int, context: int = 768, groups: int = 32):
        channels = num_heads * dims
        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.layer_norm_1 = nn.LayerNorm(channels)
        self.attention_1 = nn.MultiheadAttention(
            channels, num_heads, batch_first=True, bias=False
        )
        self.layer_norm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttentionBlock(
            num_heads, channels, context, in_proj_bias=False
        )
        self.layer_norm_3 = nn.LayerNorm(channels)
        self.linear_gelu_1 = nn.Linear(channels, 4 * 2 * channels)
        self.linear_gelu_2 = nn.Linear(4 * channels, 2 * channels)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        image_shape = x.shape[-2:]
        h = self.group_norm(x)
        h = self.conv(h)
        h_atn = reshape_for_attention(h)
        h_atn = self.layer_norm_1(h_atn)
        h_atn = self.attention_1(h_atn, h_atn, h_atn)[0]
        h += h_atn

        h_atn = self.layer_norm_2(h)
        h_atn = self.attention_2(h_atn, context)
        h += h_atn

        h_proj = self.layer_norm_3(h)
        h_proj, gate = self.linear_gelu_1(h_proj).chunk(2, dim=-1)
        h_proj = h_proj * F.gelu(gate)
        h_proj = self.linear_gelu_2(h_proj)
        h += h_proj
        h = reshape_for_attention(h, image_shape)
        h = self.conv_out(h)
        h += x
        return h


class UNetResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_timesteps: int = 1280,
        groups: int = 32,
    ):
        super().__init__()
        self.group_norm_1 = nn.GroupNorm(groups, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_timesteps, out_channels)

        self.group_norm_2 = nn.GroupNorm(groups, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, t):

        h = self.group_norm_1(h)
        h = F.silu(h)
        h = self.conv_1(h)
        h = self.linear_time(F.silu(t))[:, :, None, None] + h

        h = self.group_norm_2(h)
        h = F.silu(h)
        h = self.conv_2(h)
        h = h + self.skip(x)

        return h


class UNet(nn.Module):

    def __init__(
        self,
        base_dims: int = 320,
        base_attention_dims: int = 40,
        num_heads: int = 8,
        in_dims: int = 4,
        out_groups: int = 32,
        out_dims: int = 4,
    ):
        super().__init__()

        self.encoder = nn.ModuleList(
            [  # f=8
                SwitchSequential(
                    nn.Conv2d(in_dims, base_dims, kernel_size=3, padding=1)
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims, base_dims),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims, base_dims),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims),
                ),
                # f=16
                SwitchSequential(
                    nn.Conv2d(base_dims, base_dims, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims, base_dims * 2),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 2),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 2, base_dims * 2),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 2),
                ),
                # f=32
                SwitchSequential(
                    nn.Conv2d(
                        base_dims * 2, base_dims * 2, kernel_size=3, stride=2, padding=1
                    )
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 2, base_dims * 4),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 4),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 4, base_dims * 4),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 4),
                ),
                # f=64
                SwitchSequential(
                    nn.Conv2d(
                        base_dims * 4, base_dims * 4, kernel_size=3, stride=2, padding=1
                    )
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 4, base_dims * 4),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 4, base_dims * 4),
                ),
            ]
        )

        self.bottle_neck = SwitchSequential(
            UNetResidualBlock(base_dims * 4, base_dims * 4),
            UNetCrossAttentionBlock(num_heads, base_attention_dims * 4),
            UNetResidualBlock(base_dims * 4, base_dims * 4),
        )

        # decoder receive double number of channels as input because of the skip connections
        self.decoder = nn.ModuleList(
            [
                # f=32
                SwitchSequential(
                    UNetResidualBlock(base_dims * 8, base_dims * 4),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 8, base_dims * 4),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 8, base_dims * 4),
                    UpScale(base_dims * 4, base_dims * 4),
                ),
                # f=16
                SwitchSequential(
                    UNetResidualBlock(base_dims * 8, base_dims * 4),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 4),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 8, base_dims * 4),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 4),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 6, base_dims * 4),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 4),
                    UpScale(base_dims * 4, base_dims * 4),
                ),
                # f=8
                SwitchSequential(
                    UNetResidualBlock(base_dims * 6, base_dims * 2),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 2),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 4, base_dims * 2),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 2),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 3, base_dims * 2),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 2),
                    UpScale(base_dims * 2, base_dims * 2),
                ),
                # f=4
                SwitchSequential(
                    UNetResidualBlock(base_dims * 3, base_dims),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 2, base_dims),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 2),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 2, base_dims),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims),
                ),
            ]
        )

        self.output = nn.Sequential(
            nn.GroupNorm(out_groups, base_dims),
            nn.SiLU(),
            nn.Conv2d(base_dims, out_dims, kernel_size=3, padding=1),
        )

    def forward(self, x, context, t):

        out = self.output(x)
