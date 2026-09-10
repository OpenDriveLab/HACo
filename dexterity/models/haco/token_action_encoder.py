"""Per-token action encoder used by trained real-time chunking."""

from __future__ import annotations

import torch

from gr00t.model.modules.embodiment_conditioned_mlp import (
    MultiEmbodimentActionEncoder,
    swish,
)


class PerTokenActionEncoder(MultiEmbodimentActionEncoder):
    """Action encoder with optional per-action-token timesteps."""

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        if timesteps.ndim == 1 and timesteps.shape == (batch_size,):
            timesteps = timesteps[:, None].expand(-1, horizon)
        elif timesteps.ndim != 2 or timesteps.shape != (batch_size, horizon):
            raise ValueError(
                "timesteps must have shape (B,) or (B,T); got "
                f"{tuple(timesteps.shape)} for actions {tuple(actions.shape)}"
            )
        action_embedding = self.W1(actions, cat_ids)
        time_embedding = self.pos_encoding(timesteps).to(dtype=action_embedding.dtype)
        hidden = swish(
            self.W2(torch.cat((action_embedding, time_embedding), dim=-1), cat_ids)
        )
        return self.W3(hidden, cat_ids)


__all__ = ["PerTokenActionEncoder"]
