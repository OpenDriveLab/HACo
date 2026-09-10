from __future__ import annotations

import torch
from torch import nn

from .sensor_common import FingerTokenContextualizer, make_mlp


# Joint indices are root-to-tip inside each finger; outer order matches the
# physical/tactile wire contract shared by all PACE encoders.
FINGER_JOINT_INDICES = (
    (30, 31, 32, 33, 34),
    (35, 36, 37, 38),
    (26, 27, 28, 29),
    (22, 23, 24, 25),
    (39, 40, 41, 42, 43),
    (8, 9, 10, 11, 12),
    (13, 14, 15, 16),
    (4, 5, 6, 7),
    (0, 1, 2, 3),
    (17, 18, 19, 20, 21),
)
FINGER_TYPES = (
    "pinky", "ring", "middle", "index", "thumb",
    "pinky", "ring", "middle", "index", "thumb",
)
FINGER_DOF = {"pinky": 5, "ring": 4, "middle": 4, "index": 4, "thumb": 5}


class ForceFingerEncoder(nn.Module):
    """Encode 9-frame measured-torque histories into ten local finger tokens."""

    def __init__(self, config) -> None:
        super().__init__()
        self.history_length = int(config.force_history_length)
        self.joint_count = int(config.force_joint_count)
        if self.joint_count != 44:
            raise ValueError(f"Sharpa force encoder requires 44 joints, got {self.joint_count}")
        self.encoder_dim = int(config.sensor_encoder_dim)
        self.joint_dim = int(getattr(config, "joint_history_dim", 64))
        self.joint_history_mlp = make_mlp(
            self.history_length * 2, self.joint_dim, self.joint_dim
        )
        self.finger_mlps = nn.ModuleDict(
            {
                name: make_mlp(
                    FINGER_DOF[name] * self.joint_dim + FINGER_DOF[name],
                    self.encoder_dim,
                    self.encoder_dim,
                )
                for name in FINGER_DOF
            }
        )

    def forward(
        self, history: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.joint_count, self.history_length)
        if tuple(history.shape[-2:]) != expected:
            raise ValueError(
                f"force_history must end in {expected}, got {tuple(history.shape)}"
            )
        if valid.shape != history.shape:
            raise ValueError(
                "force_history_valid must match force_history, got "
                f"{tuple(valid.shape)} and {tuple(history.shape)}"
            )
        valid = valid.bool() & torch.isfinite(history)
        clean = torch.where(valid, history, torch.zeros_like(history))
        joint_input = torch.cat((clean, valid.to(clean.dtype)), dim=-1)
        joint_features = self.joint_history_mlp(joint_input)
        joint_available = valid.any(dim=-1)
        joint_features = joint_features * joint_available.unsqueeze(-1).to(
            joint_features.dtype
        )

        local_tokens = []
        finger_valid = []
        for finger_type, indices in zip(FINGER_TYPES, FINGER_JOINT_INDICES):
            index = torch.tensor(indices, device=history.device)
            features = joint_features.index_select(1, index).flatten(start_dim=1)
            available = joint_available.index_select(1, index)
            finger_input = torch.cat((features, available.to(features.dtype)), dim=-1)
            token = self.finger_mlps[finger_type](finger_input)
            token_valid = available.any(dim=-1)
            token = token * token_valid.unsqueeze(-1).to(token.dtype)
            local_tokens.append(token)
            finger_valid.append(token_valid)
        return torch.stack(local_tokens, dim=1), torch.stack(finger_valid, dim=1)


class ForceEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.local_encoder = ForceFingerEncoder(config)
        self.contextualizer = FingerTokenContextualizer(
            config, add_modality_embedding=True
        )

    def encode_local(self, history, valid):
        return self.local_encoder(history, valid)

    def forward(self, history, valid):
        return self.contextualizer(*self.encode_local(history, valid))
