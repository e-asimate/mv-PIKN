"""Layers for the mv-PIKN training pipeline."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn






def make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float = 0.0,
    activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    act = activation if activation is not None else nn.GELU()
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        act,
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class RawEncoder(nn.Module):
    def __init__(self, n_channels: int, seq_len: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=7, padding=3, groups=n_channels, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        mid = max(32, n_channels * 2)
        self.pointwise = nn.Sequential(
            nn.Conv1d(n_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Conv1d(mid, mid, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(mid * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = torch.cat([x.mean(dim=-1), x.max(dim=-1).values], dim=-1)
        return self.norm(self.proj(x))


class SpectralEncoder(nn.Module):
    def __init__(self, n_channels: int, n_freqs: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = make_mlp(n_channels * n_freqs, hidden_dim, out_dim, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)
        return self.norm(self.net(x))


class FusionModule(nn.Module):
    def __init__(self, stream_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.Linear(stream_dim, 1, bias=False)
        self.proj = make_mlp(stream_dim, out_dim, out_dim, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(streams), dim=1)
        fused = (weights * streams).sum(dim=1)
        return self.norm(self.proj(fused))


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = make_mlp(latent_dim, latent_dim * 2, out_dim, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
