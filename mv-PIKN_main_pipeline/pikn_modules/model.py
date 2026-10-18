"""Model for the mv-PIKN training pipeline."""

from __future__ import annotations

from typing import Any
from typing import Dict
import torch
import torch.nn as nn



from .dynamics import (KoopmanOperator, WilsonCowanCell)
from .layers import (Decoder, FusionModule, RawEncoder, SpectralEncoder)


class TransitionPIKN(nn.Module):
    def __init__(
        self,
        raw_past_channels: int,
        raw_channels: int,
        seq_len: int,
        spec_in_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        spectral_hidden_dim: int = 128,
        temp_dropout: float = 0.1,
        fusion_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
        koopman_rank: int = 8,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        raw_enc_out = hidden_dim
        self.raw_encoder = RawEncoder(raw_channels, seq_len, raw_enc_out, dropout=temp_dropout)
        self.raw_past_encoder = (
            self.raw_encoder if raw_past_channels == raw_channels
            else RawEncoder(raw_past_channels, seq_len, raw_enc_out, dropout=temp_dropout)
        )
        self.spec_encoder = SpectralEncoder(1, spec_in_dim, spectral_hidden_dim, spectral_hidden_dim, dropout=temp_dropout)

        self.proj_raw_past = nn.Linear(raw_enc_out, latent_dim)
        self.proj_raw_t = nn.Linear(raw_enc_out, latent_dim)
        self.proj_raw_tdt = nn.Linear(raw_enc_out, latent_dim)
        self.proj_spec_t = nn.Linear(spectral_hidden_dim, latent_dim)
        self.proj_spec_tdt = nn.Linear(spectral_hidden_dim, latent_dim)

        self.koopman = KoopmanOperator(latent_dim, rank=koopman_rank)
        self.wc_cell = WilsonCowanCell(latent_dim, dt=dt)
        self.decoder = Decoder(latent_dim, raw_enc_out, dropout=decoder_dropout)
        self.fusion = FusionModule(latent_dim, hidden_dim, dropout=fusion_dropout)

        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.ictal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        raw_past_feat = self.raw_past_encoder(batch["raw_past"])
        raw_t_feat = self.raw_encoder(batch["raw_t"])
        raw_tdt_feat = self.raw_encoder(batch["raw_tdt"])
        spec_t_feat = self.spec_encoder(batch["spec_t"].reshape(batch["spec_t"].shape[0], -1))
        spec_tdt_feat = self.spec_encoder(batch["spec_tdt"].reshape(batch["spec_tdt"].shape[0], -1))

        z_past = self.proj_raw_past(raw_past_feat)
        z_raw_t = self.proj_raw_t(raw_t_feat)
        z_raw_tdt = self.proj_raw_tdt(raw_tdt_feat)
        z_spec_t = self.proj_spec_t(spec_t_feat)
        z_spec_tdt = self.proj_spec_tdt(spec_tdt_feat)

        z_koopman_pred = self.koopman(z_raw_t)
        z_wc_pred = self.wc_cell(z_raw_t)
        z_recon = self.decoder(z_raw_t)

        fused = self.fusion(torch.stack([z_past, z_raw_t, z_spec_t], dim=1))
        risk_logit = self.risk_head(fused).squeeze(-1)
        ictal_logit = self.ictal_head(fused).squeeze(-1)

        return {
            "risk_logit": risk_logit,
            "ictal_logit": ictal_logit,
            "z_raw_t": z_raw_t,
            "z_raw_tdt": z_raw_tdt,
            "z_spec_t": z_spec_t,
            "z_spec_tdt": z_spec_tdt,
            "z_koopman_pred": z_koopman_pred,
            "z_wc_pred": z_wc_pred,
            "z_recon": z_recon,
            "z_raw_t_enc": raw_t_feat,
            "fused": fused,
        }
