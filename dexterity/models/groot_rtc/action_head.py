"""Trained hard-prefix real-time chunking action head for GR00T N1.7."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from dexterity.models.groot_n17.model import GrootN17ActionHead
from gr00t.model.modules.embodiment_conditioned_mlp import (
    MultiEmbodimentActionEncoder,
    swish,
)

from .config import GrootRTCConfig


class PerTokenActionEncoder(MultiEmbodimentActionEncoder):
    """The stock action encoder with optional per-action-token timesteps."""

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        if timesteps.ndim == 1 and timesteps.shape == (batch_size,):
            timesteps = timesteps[:, None].expand(-1, horizon)
        elif timesteps.ndim == 2 and timesteps.shape == (batch_size, horizon):
            pass
        else:
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


class GrootRTCActionHead(GrootN17ActionHead):
    """GR00T flow head trained with clean action prefixes and postfix-only loss."""

    def __init__(self, config: GrootRTCConfig) -> None:
        super().__init__(config)
        # Same parameter names/shapes as upstream; only forward semantics differ.
        self.action_encoder = PerTokenActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.set_trainable_parameters(
            config.tune_projector,
            config.tune_diffusion_model,
            config.tune_vlln,
        )
        self.rtc_prefix_weights = tuple(
            float(value) for value in config.rtc_training_prefix_weights
        )

    def _sample_prefix_steps(self, action_mask: torch.Tensor) -> torch.Tensor:
        batch_size = action_mask.shape[0]
        weights = torch.as_tensor(
            self.rtc_prefix_weights,
            dtype=torch.float32,
            device=action_mask.device,
        )
        weights = weights[None, :].expand(batch_size, -1).clone()
        valid_steps = action_mask.bool().any(dim=-1).sum(dim=-1)
        candidates = torch.arange(weights.shape[1], device=weights.device)[None, :]
        weights.masked_fill_(candidates >= valid_steps[:, None], 0.0)
        if torch.any(weights.sum(dim=-1) <= 0):
            raise ValueError(
                "every RTC training sample must contain at least one valid action"
            )
        return torch.multinomial(weights, num_samples=1, replacement=True).squeeze(1)

    def _action_features(
        self,
        actions: torch.Tensor,
        token_timesteps: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        features = self.action_encoder(actions, token_timesteps, embodiment_id)
        if self.config.add_pos_embed:
            positions = torch.arange(features.shape[1], device=features.device)
            features = features + self.position_embedding(positions)[None]
        return features

    def _predict_velocity(
        self,
        *,
        vl_embeds: torch.Tensor,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
        global_timestep: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        return_all_hidden_states: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        state_action_embeddings = torch.cat((state_features, action_features), dim=1)
        kwargs: dict[str, Any] = {
            "hidden_states": state_action_embeddings,
            "encoder_hidden_states": vl_embeds,
            "timestep": global_timestep,
        }
        if return_all_hidden_states:
            kwargs.update(
                {
                    "encoder_attention_mask": backbone_output.backbone_attention_mask,
                    "return_all_hidden_states": True,
                }
            )
        if self.config.use_alternate_vl_dit:
            kwargs.update(
                {
                    "image_mask": backbone_output.image_mask,
                    "backbone_attention_mask": backbone_output.backbone_attention_mask,
                }
            )
        output = self.model(**kwargs)
        if return_all_hidden_states:
            model_output, hidden_states = output
        else:
            model_output, hidden_states = output, None
        prediction = self.action_decoder(model_output, embodiment_id)
        return prediction[:, -action_features.shape[1] :], hidden_states

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        state = action_input.state
        if state.shape[1] != self.config.state_history_length:
            raise ValueError("current state history does not match checkpoint config")
        state = state.view(state.shape[0], 1, -1)
        state_features = self.state_encoder(state, embodiment_id)
        if self.training and self.state_dropout_prob > 0:
            drop = torch.rand(state_features.shape[0], device=state_features.device)
            drop = (drop < self.state_dropout_prob)[:, None, None]
            state_features = state_features * (~drop).to(state_features.dtype)

        actions = action_input.action
        action_mask = action_input.action_mask
        noise = torch.randn_like(actions)
        flow_time = self.sample_time(
            actions.shape[0], device=actions.device, dtype=actions.dtype
        )
        prefix_steps = self._sample_prefix_steps(action_mask)
        positions = torch.arange(actions.shape[1], device=actions.device)[None, :]
        prefix_mask = positions < prefix_steps[:, None]
        token_time = flow_time[:, None].expand(-1, actions.shape[1])
        token_time = torch.where(prefix_mask, torch.ones_like(token_time), token_time)
        noisy_actions = (
            token_time[..., None] * actions
            + (1.0 - token_time[..., None]) * noise
        )
        velocity_target = actions - noise

        token_timesteps = (token_time * self.num_timestep_buckets).long()
        global_timestep = (flow_time * self.num_timestep_buckets).long()
        action_features = self._action_features(
            noisy_actions, token_timesteps, embodiment_id
        )
        predicted_velocity, hidden_states = self._predict_velocity(
            vl_embeds=vl_embeds,
            state_features=state_features,
            action_features=action_features,
            global_timestep=global_timestep,
            embodiment_id=embodiment_id,
            backbone_output=backbone_output,
            return_all_hidden_states=True,
        )

        rtc_action_mask = action_mask * (~prefix_mask)[..., None].to(action_mask.dtype)
        action_loss = (
            F.mse_loss(predicted_velocity, velocity_target, reduction="none")
            * rtc_action_mask
        )
        loss = action_loss.sum() / (rtc_action_mask.sum() + 1e-6)
        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": rtc_action_mask,
            "rtc_prefix_steps": prefix_steps,
            "backbone_features": vl_embeds,
            "state_features": state_features,
            "all_hidden_states": hidden_states,
        }

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        prefix_steps = int((options or {}).get("rtc_prefix_steps", 0))
        if not 0 <= prefix_steps <= self.config.rtc_training_max_prefix_steps:
            raise ValueError(
                f"rtc_prefix_steps={prefix_steps} exceeds checkpoint capability "
                f"0..{self.config.rtc_training_max_prefix_steps}"
            )
        if prefix_steps and "action" not in action_input:
            raise ValueError(
                "trained RTC inference requires a normalized action prefix"
            )

        batch_size = backbone_features.shape[0]
        actions = torch.randn(
            (batch_size, self.action_horizon, self.action_dim),
            dtype=backbone_features.dtype,
            device=backbone_features.device,
        )
        prefix = None
        if prefix_steps:
            prefix = action_input.action[:, :prefix_steps].clone()
            actions[:, :prefix_steps] = prefix

        step_size = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            continuous_time = step / float(self.num_inference_timesteps)
            discrete_time = int(continuous_time * self.num_timestep_buckets)
            global_timestep = torch.full(
                (batch_size,), discrete_time, device=actions.device, dtype=torch.long
            )
            token_timesteps = global_timestep[:, None].expand(
                -1, self.action_horizon
            ).clone()
            if prefix_steps:
                token_timesteps[:, :prefix_steps] = self.num_timestep_buckets
                actions[:, :prefix_steps] = prefix
            action_features = self._action_features(
                actions, token_timesteps, embodiment_id
            )
            predicted_velocity, _ = self._predict_velocity(
                vl_embeds=backbone_features,
                state_features=state_features,
                action_features=action_features,
                global_timestep=global_timestep,
                embodiment_id=embodiment_id,
                backbone_output=backbone_output,
                return_all_hidden_states=False,
            )
            if prefix_steps:
                predicted_velocity[:, :prefix_steps] = 0
            actions = actions + step_size * predicted_velocity
            if prefix_steps:
                actions[:, :prefix_steps] = prefix

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": backbone_features,
                "state_features": state_features,
                "rtc_prefix_steps": torch.full(
                    (batch_size,), prefix_steps, device=actions.device, dtype=torch.long
                ),
            }
        )


__all__ = ["GrootRTCActionHead", "PerTokenActionEncoder"]
