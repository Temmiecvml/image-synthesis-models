from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.autoencoder.decoder import UpScale


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


class CrossAttention(nn.Module):
    def __init__(
        self, query_dim, num_heads=8, dim_heads=64, context_dims=None, dropout=0.0
    ):
        super().__init__()

        inner_dims = dim_heads * num_heads
        self.scale = dim_heads**-0.5
        self.num_heads = num_heads
        context_dims = context_dims or query_dim

        # its self attention if context_dims is None
        self.to_q = nn.Linear(query_dim, inner_dims, bias=False)
        self.to_k = nn.Linear(context_dims, inner_dims, bias=False)
        self.to_v = nn.Linear(context_dims, inner_dims, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dims, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        context = context or x
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.num_heads),
            (q, k, v),
        )

        w = q @ k.transpose(-2, -1) * self.scale
        w = F.softmax(w, dim=-1)
        out = w @ v
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.num_heads)
        out = self.to_out(out)

        return out


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        dim_heads: int,
        context_dims: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dims = dims
        self.dim_heads = dim_heads
        self.context_dims = context_dims

        # self attention
        self.attn1 = CrossAttention(
            query_dim=dims,
            num_heads=num_heads,
            dim_heads=dim_heads,
            context_dims=None,
            dropout=dropout,
        )

        # cross attention
        self.attn2 = CrossAttention(
            query_dim=dims,
            num_heads=num_heads,
            dim_heads=dim_heads,
            context_dims=context_dims,
            dropout=dropout,
        )

        self.ff = nn.Sequential(
            GEGLU(dims, dims * 2),
            nn.Dropout(dropout),
            nn.Linear(dims * 2, dims),
        )

        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.norm3 = nn.LayerNorm(dims)

    def forward(self, x, context):
        h = self.attn1(self.norm1(x), None) + x  # self attention
        h = self.attn2(self.norm2(h), context) + h  # cross attention
        h = self.ff(self.norm3(h)) + h
        return h


class UNetCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        context_dims: int = 640,
        depth: int = 1,
        groups: int = 32,
    ):

        self.channels = channels
        self.num_heads = num_heads
        self.context_dims = context_dims
        self.dim_heads = dim_heads = channels // num_heads
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(channels, num_heads, dim_heads, context_dims)
                for _ in range(depth)
            ]
        )

        self.proj_out = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, context):
        b, c, ih, iw = x.shape

        h = self.norm(x)
        h = self.proj_in(h)
        h = rearrange(h, "b c h w -> b (h w) c")  # flatten image dims
        for block in self.transformer_blocks:
            h = block(h, context)

        h = rearrange(h, "b (h w) c -> b c h w", h=ih, w=iw)  # unflatten image dims
        h = self.proj_out(h)
        h += x

        return h


class UNetResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dims: int = 768,
        groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(groups, in_channels, eps=1e-5, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.linear_time = nn.Linear(time_embedding_dims, out_channels)

        self.out_layers = nn.Sequential(
            nn.GroupNorm(groups, out_channels, eps=1e-5, affine=True),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x, t):
        h = self.in_layers(x)
        h = h + self.linear_time(F.silu(t))[:, :, None, None]
        h = self.out_layers(h)
        h = h + self.skip(x)

        return h


class UNet(nn.Module):

    def __init__(
        self,
        base_dims: int = 192,
        base_context_dims: int = 640,
        time_embedding_dims: int = 768,
        num_heads: int = 6,
        in_dims: int = 3,
        out_dims: int = 4,
        groups: int = 32,
    ):
        super().__init__()

        self.encoder = nn.ModuleList(
            [  # f=8
                SwitchSequential(  # 3 -> 192
                    nn.Conv2d(in_dims, base_dims, kernel_size=3, padding=1)
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims, base_dims, time_embedding_dims),
                    UNetResidualBlock(base_dims, base_dims, time_embedding_dims),
                    nn.Conv2d(base_dims, base_dims, kernel_size=3, stride=2, padding=1),
                ),
                # f=16
                SwitchSequential(
                    UNetResidualBlock(base_dims, base_dims * 2, time_embedding_dims),
                    UNetCrossAttentionBlock(
                        base_dims * 2, num_heads, base_context_dims
                    ),
                    UNetResidualBlock(
                        base_dims * 2, base_dims * 2, time_embedding_dims
                    ),
                    UNetCrossAttentionBlock(
                        base_dims * 2, num_heads, base_context_dims
                    ),
                    nn.Conv2d(
                        base_dims * 2, base_dims * 2, kernel_size=3, stride=2, padding=1
                    ),
                ),
                # f=32
                SwitchSequential(
                    UNetResidualBlock(
                        base_dims * 2, base_dims * 3, time_embedding_dims
                    ),
                    UNetCrossAttentionBlock(
                        base_dims * 3, num_heads, base_context_dims
                    ),
                    UNetResidualBlock(
                        base_dims * 3, base_dims * 3, time_embedding_dims
                    ),
                    UNetCrossAttentionBlock(
                        base_dims * 3, num_heads, base_context_dims
                    ),
                    nn.Conv2d(
                        base_dims * 3, base_dims * 3, kernel_size=3, stride=2, padding=1
                    ),
                ),
                # f=64
                SwitchSequential(
                    UNetResidualBlock(
                        base_dims * 3, base_dims * 5, time_embedding_dims
                    ),
                    UNetCrossAttentionBlock(
                        base_dims * 5, num_heads, base_context_dims
                    ),
                    UNetResidualBlock(
                        base_dims * 5, base_dims * 5, time_embedding_dims
                    ),
                    UNetCrossAttentionBlock(
                        base_dims * 5, num_heads, base_context_dims
                    ),
                ),
            ]
        )

        self.bottle_neck = SwitchSequential(
            UNetResidualBlock(base_dims * 4, base_dims * 4, time_embedding_dims),
            UNetCrossAttentionBlock(num_heads, base_attention_dims * 4),
            UNetResidualBlock(base_dims * 4, base_dims * 4, time_embedding_dims),
        )

        # decoder receive double number of channels as input because of the skip connections
        self.decoder = nn.ModuleList(
            [
                # f=32
                SwitchSequential(
                    UNetResidualBlock(
                        base_dims * 8, base_dims * 4, time_embedding_dims
                    ),
                ),
                SwitchSequential(
                    UNetResidualBlock(
                        base_dims * 8, base_dims * 4, time_embedding_dims
                    ),
                ),
                SwitchSequential(
                    UNetResidualBlock(
                        base_dims * 8, base_dims * 4, time_embedding_dims
                    ),
                    UpScale(base_dims * 4, base_dims * 4),
                ),
                # f=16
                SwitchSequential(
                    UNetResidualBlock(
                        base_dims * 8, base_dims * 4, time_embedding_dims
                    ),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 4),
                ),
                SwitchSequential(
                    UNetResidualBlock(
                        base_dims * 8, base_dims * 4, time_embedding_dims
                    ),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 4),
                ),
                SwitchSequential(
                    UNetResidualBlock(
                        base_dims * 6, base_dims * 4, time_embedding_dims
                    ),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 4),
                    UpScale(base_dims * 4, base_dims * 4),
                ),
                # f=8
                SwitchSequential(
                    UNetResidualBlock(
                        base_dims * 6, base_dims * 2, time_embedding_dims
                    ),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 2),
                ),
                SwitchSequential(
                    UNetResidualBlock(
                        base_dims * 4, base_dims * 2, time_embedding_dims
                    ),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 2),
                ),
                SwitchSequential(
                    UNetResidualBlock(
                        base_dims * 3, base_dims * 2, time_embedding_dims
                    ),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 2),
                    UpScale(base_dims * 2, base_dims * 2),
                ),
                # f=4
                SwitchSequential(
                    UNetResidualBlock(base_dims * 3, base_dims, time_embedding_dims),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 2, base_dims, time_embedding_dims),
                    UNetCrossAttentionBlock(num_heads, base_attention_dims * 2),
                ),
                SwitchSequential(
                    UNetResidualBlock(base_dims * 2, base_dims, time_embedding_dims),
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

        hs = []
        for module in self.encoder:
            h = module(x, context, t)
            hs.append(h)

        out = self.output(x)
