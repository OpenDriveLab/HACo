from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from gr00t.model.modules.dit import AlternateVLDiT


class MaskedAlternateVLDiT(AlternateVLDiT):
    """Alternate VLM cross-attention and masked suffix self-attention."""

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
    ):
        del encoder_attention_mask
        if image_mask is None or backbone_attention_mask is None:
            raise ValueError("image_mask and backbone_attention_mask are required")
        if suffix_attention_mask is not None:
            expected = hidden_states.shape[:2]
            if suffix_attention_mask.shape != expected:
                raise ValueError(
                    f"suffix_attention_mask must have shape {expected}, got "
                    f"{tuple(suffix_attention_mask.shape)}"
                )
            suffix_attention_mask = suffix_attention_mask.bool()
            query_mask = suffix_attention_mask.unsqueeze(-1)
            hidden_states = torch.where(
                query_mask, hidden_states, torch.zeros_like(hidden_states)
            )
        else:
            query_mask = None

        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        image_attention_mask = image_mask.bool() & backbone_attention_mask.bool()
        non_image_attention_mask = (~image_mask.bool()) & backbone_attention_mask.bool()
        all_hidden_states = [hidden_states]
        if not self.config.interleave_self_attention:
            raise ValueError("MaskedAlternateVLDiT requires interleave_self_attention")

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
                attend_text = block_index % (2 * self.attend_text_every_n_blocks) == 0
                cross_mask = non_image_attention_mask if attend_text else image_attention_mask
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=cross_mask,
                    temb=temb,
                )
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
