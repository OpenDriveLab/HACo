from __future__ import annotations

import torch
from torch import nn

from .force_encoder import ForceFingerEncoder
from .sensor_common import FingerTokenContextualizer, make_mlp
from .tactile_encoder import TactileFingerEncoder


class ForceTactileEncoder(nn.Module):
    """Fuse same-finger force/tactile features before one ten-finger attention."""

    def __init__(self, config) -> None:
        super().__init__()
        self.encoder_dim = int(config.sensor_encoder_dim)
        self.force_local_encoder = ForceFingerEncoder(config)
        self.tactile_local_encoder = TactileFingerEncoder(config)
        self.fusion_mlp = make_mlp(
            self.encoder_dim * 2 + 2, self.encoder_dim * 2, self.encoder_dim
        )
        self.contextualizer = FingerTokenContextualizer(
            config, add_modality_embedding=False
        )

    def forward(
        self,
        force_history,
        force_valid,
        wrench_history,
        wrench_valid,
        deformation,
        deformation_valid,
    ):
        force_tokens, force_mask = self.force_local_encoder(force_history, force_valid)
        tactile_tokens, tactile_mask = self.tactile_local_encoder(
            wrench_history, wrench_valid, deformation, deformation_valid
        )
        force_tokens = force_tokens * force_mask.unsqueeze(-1).to(force_tokens.dtype)
        tactile_tokens = tactile_tokens * tactile_mask.unsqueeze(-1).to(
            tactile_tokens.dtype
        )
        fusion_input = torch.cat(
            (
                force_tokens,
                tactile_tokens,
                force_mask.unsqueeze(-1).to(force_tokens.dtype),
                tactile_mask.unsqueeze(-1).to(force_tokens.dtype),
            ),
            dim=-1,
        )
        fused_tokens = self.fusion_mlp(fusion_input)
        fused_mask = force_mask | tactile_mask
        fused_tokens = fused_tokens * fused_mask.unsqueeze(-1).to(fused_tokens.dtype)
        return self.contextualizer(fused_tokens, fused_mask)
