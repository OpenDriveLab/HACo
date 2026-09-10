"""Dedicated physical cross-attention routes for HACO."""

from __future__ import annotations

import torch
from torch import nn
from diffusers.models.attention import Attention

from gr00t.model.modules.dit import AdaLayerNorm, _sdpa_context

from dexterity.models.pace.masked_dit import MaskedAlternateVLDiT
from dexterity.models.pace.variants.fuse_v2_physcross.physical_dit import (
    GatedPhysicalCrossAttention,
    PhysicalCrossMaskedAlternateVLDiT,
)


class UngatedPhysicalCrossAttention(nn.Module):
    """Physical cross-attention with an unconditional residual connection."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        attention_bias: bool,
        upcast_attention: bool,
        attention_out_bias: bool,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm = AdaLayerNorm(
            dim, norm_elementwise_affine=False, norm_eps=norm_eps
        )
        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        physical_hidden_states: torch.Tensor,
        physical_attention_mask: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        query = self.norm(hidden_states, temb)
        with _sdpa_context():
            update = self.attention(
                query,
                encoder_hidden_states=physical_hidden_states,
                attention_mask=physical_attention_mask,
            )
        return hidden_states + update


class HacoPhysicalCrossDiT(PhysicalCrossMaskedAlternateVLDiT):
    """The one PFI cross route, configured as genuinely gated or ungated."""

    def __init__(self, *args, gated: bool, gate_init: float = 0.0, **kwargs) -> None:
        super().__init__(*args, physical_cross_gate_init=gate_init, **kwargs)
        self.gated = bool(gated)
        if not self.gated:
            self.physical_cross_adapters = nn.ModuleList(
                [
                    UngatedPhysicalCrossAttention(
                        dim=self.inner_dim,
                        heads=int(self.config.num_attention_heads),
                        dim_head=int(self.config.attention_head_dim),
                        dropout=float(self.config.dropout),
                        attention_bias=bool(self.config.attention_bias),
                        upcast_attention=bool(self.config.upcast_attention),
                        attention_out_bias=True,
                        norm_eps=float(self.config.norm_eps),
                    )
                    for _ in self.physical_cross_block_indices
                ]
            )

    @property
    def uses_learned_gate(self) -> bool:
        return all(
            isinstance(adapter, GatedPhysicalCrossAttention)
            for adapter in self.physical_cross_adapters
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep=None,
        encoder_attention_mask=None,
        return_all_hidden_states: bool = False,
        image_mask=None,
        backbone_attention_mask=None,
        suffix_attention_mask=None,
        physical_hidden_states=None,
        physical_attention_mask=None,
    ):
        if physical_hidden_states is None or physical_attention_mask is None:
            raise ValueError("physical hidden states and mask are required")
        if physical_hidden_states.shape[1] == 0:
            # hp_wo_haptic keeps the full model's integration route while
            # removing physical observations.  Empty memory is a strict no-op:
            # no null token and no physical adapter participates in the graph.
            return MaskedAlternateVLDiT.forward(
                self,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                return_all_hidden_states=return_all_hidden_states,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                suffix_attention_mask=suffix_attention_mask,
            )
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            return_all_hidden_states=return_all_hidden_states,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
            suffix_attention_mask=suffix_attention_mask,
            physical_hidden_states=physical_hidden_states,
            physical_attention_mask=physical_attention_mask,
        )


__all__ = [
    "HacoPhysicalCrossDiT",
    "UngatedPhysicalCrossAttention",
]
