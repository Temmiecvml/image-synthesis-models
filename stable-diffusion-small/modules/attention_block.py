import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_attention(q, k, v):
    # Scale by sqrt(head_channels)
    scale_factor = q.shape[-1] ** -0.5
    # q: (batch_size * num_heads, seq_len, head_channels)
    # k: (batch_size * num_heads, head_channels, seq_len)
    attn = (
        torch.bmm(q, k.transpose(-1, -2)) * scale_factor
    )  # (batch_size * num_heads, seq_len, seq_len)

    # Apply softmax to the attention scores
    attn = F.softmax(attn, dim=-1)

    # Compute output by applying attention to the values
    # v: (batch_size * num_heads, seq_len, head_channels)
    x = torch.bmm(attn, v)  # (batch_size * num_heads, seq_len, head_channels)

    return x, attn


class VAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=4,
        num_groups=32,
    ):
        super(VAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.head_channels = in_channels // num_heads
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, padding=0)
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        batch_size, _, height, width = x.shape

        qkv = self.qkv(x)  # (batch_size, 3 * in_channels, height, width)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape q, k, v to (batch_size * num_heads, seq_len, head_channels)
        q = (
            q.reshape(batch_size, self.num_heads, self.head_channels, height * width)
            .permute(0, 1, 3, 2)
            .reshape(-1, height * width, self.head_channels)
        )
        k = (
            k.reshape(batch_size, self.num_heads, self.head_channels, height * width)
            .permute(0, 1, 3, 2)
            .reshape(-1, height * width, self.head_channels)
        )
        v = (
            v.reshape(batch_size, self.num_heads, self.head_channels, height * width)
            .permute(0, 1, 3, 2)
            .reshape(-1, height * width, self.head_channels)
        )

        x, _ = compute_attention(q, k, v)

        # Reshape x back to (batch_size, num_heads, head_channels, height, width)
        x = (
            x.reshape(batch_size, self.num_heads, height * width, self.head_channels)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, self.in_channels, height, width)
        )
        x = x + residual
        x = self.out(x)

        return x
