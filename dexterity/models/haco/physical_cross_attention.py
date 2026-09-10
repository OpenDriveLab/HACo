from __future__ import annotations

from typing import Optional

from diffusers.models.attention import Attention
import torch
from torch import nn
import torch.nn.functional as F

from gr00t.model.modules.dit import AdaLayerNorm, _sdpa_context

from dexterity.models.haco.masked_dit import MaskedAlternateVLDiT


class GatedPhysicalCrossAttention(nn.Module):
    """Attention-only physical adapter with a zero-safe residual gate."""

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
        gate_init: float,
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
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

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
        return hidden_states + torch.tanh(self.gate).to(update.dtype) * update


class PhysicalCrossMaskedAlternateVLDiT(MaskedAlternateVLDiT):
    """Text/image alternation plus a third, dedicated physical memory route.

    A lightweight physical cross adapter is inserted after every image-cross
    block (base indices 2, 6, ..., 30), immediately before the following suffix
    self-attention block.  The original DiT block count and weights stay intact.
    """

    def __init__(
        self,
        *args,
        physical_cross_gate_init: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.physical_cross_block_indices = tuple(
            index for index in range(len(self.transformer_blocks)) if index % 4 == 2
        )
        self.physical_cross_adapters = nn.ModuleList(
            [
                GatedPhysicalCrossAttention(
                    dim=self.inner_dim,
                    heads=int(self.config.num_attention_heads),
                    dim_head=int(self.config.attention_head_dim),
                    dropout=float(self.config.dropout),
                    attention_bias=bool(self.config.attention_bias),
                    upcast_attention=bool(self.config.upcast_attention),
                    attention_out_bias=True,
                    norm_eps=float(self.config.norm_eps),
                    gate_init=physical_cross_gate_init,
                )
                for _ in self.physical_cross_block_indices
            ]
        )
        # Used only for a sample whose ten physical observations are all
        # invalid, preventing an all-masked KV row in scaled-dot attention.
        self.null_physical_token = nn.Parameter(
            torch.zeros(1, 1, self.inner_dim)
        )

    def _safe_physical_memory(
        self,
        physical_hidden_states: torch.Tensor,
        physical_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_invalid = ~physical_attention_mask.any(dim=-1, keepdim=True)
        null_token = self.null_physical_token.to(
            device=physical_hidden_states.device,
            dtype=physical_hidden_states.dtype,
        ).expand(physical_hidden_states.shape[0], -1, -1)
        return (
            torch.cat((physical_hidden_states, null_token), dim=1),
            torch.cat((physical_attention_mask, all_invalid), dim=1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
        suffix_attention_mask: Optional[torch.Tensor] = None,
        physical_hidden_states: Optional[torch.Tensor] = None,
        physical_attention_mask: Optional[torch.Tensor] = None,
    ):
        del encoder_attention_mask
        if image_mask is None or backbone_attention_mask is None:
            raise ValueError("image_mask and backbone_attention_mask are required")
        if physical_hidden_states is None or physical_attention_mask is None:
            raise ValueError("physical hidden states and mask are required")
        if physical_hidden_states.ndim != 3:
            raise ValueError("physical_hidden_states must be [B,P,D]")
        expected_physical = physical_hidden_states.shape[:2]
        if tuple(physical_attention_mask.shape) != expected_physical:
            raise ValueError(
                "physical_attention_mask must match physical tokens, got "
                f"{tuple(physical_attention_mask.shape)} and {expected_physical}"
            )
        if physical_hidden_states.shape[-1] != self.inner_dim:
            raise ValueError(
                f"physical token width must be {self.inner_dim}, got "
                f"{physical_hidden_states.shape[-1]}"
            )
        if suffix_attention_mask is not None:
            expected_suffix = hidden_states.shape[:2]
            if tuple(suffix_attention_mask.shape) != expected_suffix:
                raise ValueError(
                    f"suffix_attention_mask must have shape {expected_suffix}, got "
                    f"{tuple(suffix_attention_mask.shape)}"
                )
            suffix_attention_mask = suffix_attention_mask.bool()
            query_mask = suffix_attention_mask.unsqueeze(-1)
            hidden_states = torch.where(
                query_mask, hidden_states, torch.zeros_like(hidden_states)
            )
        else:
            query_mask = None

        physical_attention_mask = physical_attention_mask.bool()
        physical_hidden_states = physical_hidden_states.contiguous()
        physical_hidden_states, physical_attention_mask = self._safe_physical_memory(
            physical_hidden_states, physical_attention_mask
        )
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        image_attention_mask = image_mask.bool() & backbone_attention_mask.bool()
        text_attention_mask = (~image_mask.bool()) & backbone_attention_mask.bool()
        all_hidden_states = [hidden_states]
        if not self.config.interleave_self_attention:
            raise ValueError(
                "PhysicalCrossMaskedAlternateVLDiT requires interleaved self attention"
            )

        adapter_index = 0
        for block_index, block in enumerate(self.transformer_blocks):
            if block_index % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    attention_mask=suffix_attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                attend_text = (
                    block_index % (2 * self.attend_text_every_n_blocks) == 0
                )
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=(
                        text_attention_mask if attend_text else image_attention_mask
                    ),
                    temb=temb,
                )
            if block_index in self.physical_cross_block_indices:
                hidden_states = self.physical_cross_adapters[adapter_index](
                    hidden_states,
                    physical_hidden_states,
                    physical_attention_mask,
                    temb,
                )
                adapter_index += 1
            if query_mask is not None:
                hidden_states = torch.where(
                    query_mask, hidden_states, torch.zeros_like(hidden_states)
                )
            all_hidden_states.append(hidden_states)

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        output = self.proj_out_2(hidden_states)
        if query_mask is not None:
            output = torch.where(query_mask, output, torch.zeros_like(output))
        if return_all_hidden_states:
            return output, all_hidden_states
        return output
