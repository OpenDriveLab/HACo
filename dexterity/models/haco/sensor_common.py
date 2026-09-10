from __future__ import annotations

import torch
from torch import nn


FINGER_NAMES = (
    "right_pinky",
    "right_ring",
    "right_middle",
    "right_index",
    "right_thumb",
    "left_pinky",
    "left_ring",
    "left_middle",
    "left_index",
    "left_thumb",
)
HAND_IDS = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
DIGIT_IDS = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4)


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class FingerTokenContextualizer(nn.Module):
    """Attach physical identity, contextualize ten fingers, and project to DiT."""

    def __init__(self, config, *, add_modality_embedding: bool) -> None:
        super().__init__()
        self.finger_count = int(config.tactile_finger_count)
        if self.finger_count != len(FINGER_NAMES):
            raise ValueError(
                f"finger contextualizer requires {len(FINGER_NAMES)} fingers, "
                f"got {self.finger_count}"
            )
        self.encoder_dim = int(config.sensor_encoder_dim)
        self.output_dim = int(config.input_embedding_dim)
        self.hand_embedding = nn.Embedding(2, self.encoder_dim)
        self.digit_embedding = nn.Embedding(5, self.encoder_dim)
        self.modality_embedding = (
            nn.Parameter(torch.zeros(1, 1, self.encoder_dim))
            if add_modality_embedding
            else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim,
            nhead=int(config.sensor_encoder_heads),
            dim_feedforward=self.encoder_dim * 4,
            dropout=float(config.attn_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.finger_encoder = nn.TransformerEncoder(
            layer, num_layers=int(config.sensor_encoder_layers)
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.encoder_dim),
            nn.Linear(self.encoder_dim, self.output_dim),
        )
        nn.init.normal_(self.hand_embedding.weight, std=0.02)
        nn.init.normal_(self.digit_embedding.weight, std=0.02)
        if self.modality_embedding is not None:
            nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(
        self, local_tokens: torch.Tensor, finger_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (local_tokens.shape[0], self.finger_count, self.encoder_dim)
        if tuple(local_tokens.shape) != expected:
            raise ValueError(
                f"local finger tokens must have shape {expected}, got "
                f"{tuple(local_tokens.shape)}"
            )
        if tuple(finger_valid.shape) != expected[:2]:
            raise ValueError(
                f"finger_valid must have shape {expected[:2]}, got "
                f"{tuple(finger_valid.shape)}"
            )
        finger_valid = finger_valid.bool()
        hand_ids = torch.tensor(HAND_IDS, device=local_tokens.device)
        digit_ids = torch.tensor(DIGIT_IDS, device=local_tokens.device)
        identity = self.hand_embedding(hand_ids) + self.digit_embedding(digit_ids)
        tokens = local_tokens + identity[None].to(local_tokens.dtype)

        safe_padding = ~finger_valid
        all_invalid = ~finger_valid.any(dim=-1)
        safe_padding = safe_padding.clone()
        safe_padding[all_invalid, 0] = False
        tokens = self.finger_encoder(tokens, src_key_padding_mask=safe_padding)
        tokens = tokens * finger_valid.unsqueeze(-1).to(tokens.dtype)
        if self.modality_embedding is not None:
            tokens = tokens + self.modality_embedding.to(tokens.dtype)
            tokens = tokens * finger_valid.unsqueeze(-1).to(tokens.dtype)
        tokens = self.output_projection(tokens)
        tokens = tokens * finger_valid.unsqueeze(-1).to(tokens.dtype)
        return tokens, finger_valid
