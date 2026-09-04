"""
Standard Softmax Transformer for benchmark comparison.

Architecture: standard multi-head attention with softmax.
Used as baseline to compare against SUBLEQ Branch Attention.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class StandardAttention(nn.Module):
    """Standard softmax attention: softmax(Q @ K^T / sqrt(d)) @ V"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape

        Q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)


class StandardAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, ff_dim: int = None, dropout: float = 0.0):
        super().__init__()
        ff_dim = ff_dim or dim * 4

        self.attn = StandardAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class StandardTransformer(nn.Module):
    """Standard transformer encoder for benchmark comparison."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            StandardAttentionBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)

        h = self.dropout(self.embedding(x) + self.pos_embedding(positions))

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        return self.output_proj(h)
