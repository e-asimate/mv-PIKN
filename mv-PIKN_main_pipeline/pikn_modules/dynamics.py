"""Dynamics for the mv-PIKN training pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F






class WilsonCowanCell(nn.Module):
    def __init__(self, latent_dim: int, dt: float = 1.0) -> None:
        super().__init__()
        self.dt = dt
        half = max(1, latent_dim // 2)
        self.half = half
        self.wee = nn.Parameter(torch.empty(half, half))
        self.wei = nn.Parameter(torch.empty(half, half))
        self.wie = nn.Parameter(torch.empty(half, half))
        self.wii = nn.Parameter(torch.empty(half, half))
        self.tau_e = nn.Parameter(torch.ones(half))
        self.tau_i = nn.Parameter(torch.ones(half))
        self.theta_e = nn.Parameter(torch.zeros(half))
        self.theta_i = nn.Parameter(torch.zeros(half))
        for w in (self.wee, self.wei, self.wie, self.wii):
            nn.init.normal_(w, mean=0.0, std=0.05)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        E = z[:, : self.half]
        I = z[:, self.half : self.half * 2]
        tau_e = F.softplus(self.tau_e) + 1e-3
        tau_i = F.softplus(self.tau_i) + 1e-3
        x_e = E @ self.wee.T - I @ self.wei.T + self.theta_e
        x_i = E @ self.wie.T - I @ self.wii.T + self.theta_i
        dE = (-E + torch.sigmoid(x_e)) / tau_e
        dI = (-I + torch.sigmoid(x_i)) / tau_i
        E_next = E + self.dt * dE
        I_next = I + self.dt * dI
        rest = z[:, self.half * 2:]
        return torch.cat([E_next, I_next, rest], dim=1)


class KoopmanOperator(nn.Module):
    def __init__(self, latent_dim: int, rank: int = 8) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.empty(latent_dim, rank))
        self.V = nn.Parameter(torch.empty(latent_dim, rank))
        nn.init.normal_(self.U, std=0.01)
        nn.init.normal_(self.V, std=0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        K = torch.eye(z.shape[-1], device=z.device, dtype=z.dtype) + self.U @ self.V.T
        return z @ K.T

    def stability_loss(self, z: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self(z).norm(dim=-1), z.norm(dim=-1))
