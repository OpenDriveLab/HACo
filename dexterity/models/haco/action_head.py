"""HACO flow head with joint action semantics and trained RTC."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from dexterity.models.groot_n17.model import GrootN17ActionHead
from dexterity.models.groot_rtc.action_head import PerTokenActionEncoder
from dexterity.models.pace.force_encoder import ForceEncoder
from dexterity.models.pace.force_tactile_encoder import ForceTactileEncoder
from dexterity.models.pace.masked_dit import MaskedAlternateVLDiT
from dexterity.models.pace.tactile_encoder import TactileEncoder

from .contract import get_action_contract
from .physical_integration import HacoPhysicalCrossDiT
from .rtc import (
    clamp_prefix,
    postfix_loss_mask,
    sample_prefix_steps,
    temporal_prefix_mask,
)


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    values = (prediction - target).square() * mask.to(prediction.dtype)
    return values.sum() / mask.sum().clamp_min(1).to(prediction.dtype)


class HacoActionHead(GrootN17ActionHead):
    """One configurable action head for all eleven HACO runs."""

    def __init__(self, config) -> None:
        super().__init__(config)
        if not config.use_alternate_vl_dit:
            raise ValueError("HACO requires use_alternate_vl_dit=True")
        self.contract = get_action_contract(config.action_contract)
        self.sensor_encoder_mode = str(config.sensor_encoder_mode)
        self.physical_integration = str(config.physical_integration)
        self.action_encoder = PerTokenActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        model_kwargs = dict(
            **config.diffusion_model_cfg,
            cross_attention_dim=config.backbone_embedding_dim,
            attend_text_every_n_blocks=config.attend_text_every_n_blocks,
        )
        if self.physical_integration.startswith("physcross_"):
            self.model = HacoPhysicalCrossDiT(
                **model_kwargs,
                gated=self.physical_integration == "physcross_gated",
                gate_init=float(config.physical_cross_gate_init),
            )
        else:
            self.model = MaskedAlternateVLDiT(**model_kwargs)

        self.force_encoder: nn.Module | None = None
        self.tactile_encoder: nn.Module | None = None
        self.force_tactile_encoder: nn.Module | None = None
        if self.sensor_encoder_mode == "torque_only":
            self.force_encoder = ForceEncoder(config)
        elif self.sensor_encoder_mode == "tactile_only":
            self.tactile_encoder = TactileEncoder(config)
        elif self.sensor_encoder_mode == "separate":
            self.force_encoder = ForceEncoder(config)
            self.tactile_encoder = TactileEncoder(config)
        elif self.sensor_encoder_mode == "fused":
            self.force_tactile_encoder = ForceTactileEncoder(config)
        elif self.sensor_encoder_mode != "none":
            raise ValueError(f"unsupported sensor mode {self.sensor_encoder_mode!r}")

        self.sensor_to_vlm: nn.Module | None = None
        self.physical_type_embedding: nn.Parameter | None = None
        if self.physical_integration == "imgmem":
            self.sensor_to_vlm = nn.Sequential(
                nn.LayerNorm(int(config.input_embedding_dim)),
                nn.Linear(
                    int(config.input_embedding_dim),
                    int(config.backbone_embedding_dim),
                ),
            )
            self.physical_type_embedding = nn.Parameter(
                torch.empty(1, 1, int(config.backbone_embedding_dim))
            )
            nn.init.normal_(
                self.physical_type_embedding,
                std=float(config.physical_type_embedding_std),
            )

        self.tune_sensor_encoders = bool(config.tune_sensor_encoders)
        if not self.tune_sensor_encoders:
            for module in self._sensor_modules():
                module.requires_grad_(False)
            if self.physical_type_embedding is not None:
                self.physical_type_embedding.requires_grad_(False)
        self.rtc_prefix_weights = tuple(
            float(value) for value in config.rtc_training_prefix_weights
        )
        self.set_trainable_parameters(
            config.tune_projector,
            config.tune_diffusion_model,
            config.tune_vlln,
        )
        if self.sensor_encoder_mode == "none":
            # hp_wo_haptic retains the full integration selection but its
            # physical route is structurally inactive.  Freeze unreachable
            # parameters as well, avoiding DDP unused-parameter failures.
            if isinstance(self.model, HacoPhysicalCrossDiT):
                self.model.physical_cross_adapters.requires_grad_(False)
                self.model.null_physical_token.requires_grad_(False)
            if self.sensor_to_vlm is not None:
                self.sensor_to_vlm.requires_grad_(False)
            if self.physical_type_embedding is not None:
                self.physical_type_embedding.requires_grad_(False)

    def _sensor_modules(self) -> tuple[nn.Module, ...]:
        return tuple(
            module
            for module in (
                self.force_encoder,
                self.tactile_encoder,
                self.force_tactile_encoder,
                self.sensor_to_vlm,
            )
            if module is not None
        )

    def set_frozen_modules_to_eval_mode(self):
        super().set_frozen_modules_to_eval_mode()
        if self.training and not self.tune_sensor_encoders:
            for module in self._sensor_modules():
                module.eval()

    def _encode_state(self, action_input: BatchFeature):
        state = action_input.state
        if state.shape[1] != self.config.state_history_length:
            raise ValueError("state history does not match the HACO checkpoint")
        state = state.reshape(state.shape[0], 1, -1)
        finite = torch.isfinite(state)
        valid = finite.all(dim=-1)
        state = torch.where(finite, state, torch.zeros_like(state))
        features = self.state_encoder(state, action_input.embodiment_id)
        if self.training and self.state_dropout_prob > 0:
            dropped = (
                torch.rand(features.shape[0], device=features.device)
                < self.state_dropout_prob
            )
            features = features * (~dropped)[:, None, None].to(features.dtype)
            valid = valid & (~dropped[:, None])
        return features, valid

    def _encode_sensors(self, action_input: BatchFeature, reference: torch.Tensor):
        if self.sensor_encoder_mode == "none":
            return (
                reference.new_zeros(reference.shape[0], 0, reference.shape[-1]),
                torch.zeros(
                    reference.shape[0],
                    0,
                    dtype=torch.bool,
                    device=reference.device,
                ),
            )
        if self.sensor_encoder_mode == "torque_only":
            return self.force_encoder(
                action_input.force_history, action_input.force_history_valid
            )
        if self.sensor_encoder_mode == "tactile_only":
            return self.tactile_encoder(
                action_input.tactile_wrench_history,
                action_input.tactile_wrench_valid,
                action_input.tactile_deformation,
                action_input.tactile_deformation_valid,
            )
        if self.sensor_encoder_mode == "separate":
            force, force_valid = self.force_encoder(
                action_input.force_history, action_input.force_history_valid
            )
            tactile, tactile_valid = self.tactile_encoder(
                action_input.tactile_wrench_history,
                action_input.tactile_wrench_valid,
                action_input.tactile_deformation,
                action_input.tactile_deformation_valid,
            )
            return (
                torch.stack((force, tactile), dim=2).flatten(1, 2),
                torch.stack((force_valid, tactile_valid), dim=2).flatten(1, 2),
            )
        return self.force_tactile_encoder(
            action_input.force_history,
            action_input.force_history_valid,
            action_input.tactile_wrench_history,
            action_input.tactile_wrench_valid,
            action_input.tactile_deformation,
            action_input.tactile_deformation_valid,
        )

    def _prepare_conditioning(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)
        state_features, state_valid = self._encode_state(action_input)
        sensor_features, sensor_valid = self._encode_sensors(
            action_input, state_features
        )
        sensor_features = sensor_features.to(dtype=state_features.dtype)
        backbone_features = backbone_output.backbone_features
        backbone_mask = backbone_output.backbone_attention_mask.bool()
        image_mask = backbone_output.image_mask.bool()
        if self.physical_integration == "suffix":
            fixed_features = torch.cat((state_features, sensor_features), dim=1)
            fixed_valid = torch.cat((state_valid, sensor_valid), dim=1)
        else:
            fixed_features, fixed_valid = state_features, state_valid
        if self.physical_integration == "imgmem" and sensor_features.shape[1]:
            physical_memory = self.sensor_to_vlm(sensor_features)
            physical_memory = physical_memory + self.physical_type_embedding.to(
                physical_memory.dtype
            )
            physical_memory = physical_memory * sensor_valid[..., None].to(
                physical_memory.dtype
            )
            backbone_features = torch.cat(
                (backbone_features, physical_memory), dim=1
            )
            backbone_mask = torch.cat((backbone_mask, sensor_valid), dim=1)
            image_mask = torch.cat(
                (image_mask, torch.ones_like(sensor_valid)), dim=1
            )
        routed_backbone = BatchFeature(
            data={
                **dict(backbone_output),
                "backbone_features": backbone_features,
                "backbone_attention_mask": backbone_mask,
                "image_mask": image_mask,
            }
        )
        return BatchFeature(
            data={
                "backbone_output": routed_backbone,
                "backbone_features": backbone_features,
                "state_features": state_features,
                "state_valid": state_valid,
                "sensor_features": sensor_features,
                "sensor_valid": sensor_valid,
                "fixed_features": fixed_features,
                "fixed_valid": fixed_valid,
            }
        )

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

    def _run_conditioned_model(
        self,
        suffix: torch.Tensor,
        suffix_mask: torch.Tensor,
        conditioning: BatchFeature,
        timestep: torch.Tensor,
        *,
        return_all_hidden_states: bool,
    ):
        common = dict(
            hidden_states=suffix,
            encoder_hidden_states=conditioning.backbone_features,
            timestep=timestep,
            return_all_hidden_states=return_all_hidden_states,
            image_mask=conditioning.backbone_output.image_mask,
            backbone_attention_mask=(
                conditioning.backbone_output.backbone_attention_mask
            ),
            suffix_attention_mask=suffix_mask,
        )
        if self.physical_integration.startswith("physcross_"):
            common.update(
                physical_hidden_states=conditioning.sensor_features,
                physical_attention_mask=conditioning.sensor_valid,
            )
        return self.model(**common)

    def _validate_training_action(self, action_input: BatchFeature):
        actions = action_input.action
        self.contract.validate_carrier(actions)
        action_mask = action_input.action_mask.bool()
        if action_mask.shape != actions.shape:
            raise ValueError("HACO action and mask shapes differ")
        active = self.contract.active_mask(device=actions.device).view(1, 1, -1)
        action_mask = action_mask & active
        actions = torch.where(action_mask, actions, torch.zeros_like(actions))
        return actions, action_mask

    def _component_metrics(self, prediction, target, loss_mask):
        control = loss_mask.clone()
        if self.contract.has_delta_q:
            control[..., self.contract.delta_q_slice] = False
        metrics = {
            "control_flow_mse": _masked_mse(prediction, target, control),
            "wrist_flow_mse": _masked_mse(
                prediction[..., self.contract.wrist_slice],
                target[..., self.contract.wrist_slice],
                loss_mask[..., self.contract.wrist_slice],
            ),
            "q_flow_mse": _masked_mse(
                prediction[..., self.contract.q_slice],
                target[..., self.contract.q_slice],
                loss_mask[..., self.contract.q_slice],
            ),
        }
        if self.contract.has_delta_q:
            metrics["delta_q_flow_mse"] = _masked_mse(
                prediction[..., self.contract.delta_q_slice],
                target[..., self.contract.delta_q_slice],
                loss_mask[..., self.contract.delta_q_slice],
            )
        return metrics

    def _assert_finite_training_loss(self, loss: torch.Tensor) -> None:
        """Stop immediately before Trainer can filter or checkpoint NaN/Inf."""

        if torch.isfinite(loss.detach()).all().item():
            return
        raise FloatingPointError(
            "HACO produced a non-finite training loss: "
            f"experiment={self.config.experiment_id}, "
            f"sensor={self.sensor_encoder_mode}, "
            f"integration={self.physical_integration}, "
            f"contract={self.contract.name}"
        )

    def forward(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        conditioning = self._prepare_conditioning(backbone_output, action_input)
        actions, action_mask = self._validate_training_action(action_input)
        noise = torch.randn_like(actions) * action_mask.to(actions.dtype)
        flow_time = self.sample_time(
            actions.shape[0], device=actions.device, dtype=actions.dtype
        )
        prefix_steps = sample_prefix_steps(action_mask, self.rtc_prefix_weights)
        prefix_temporal = temporal_prefix_mask(prefix_steps, actions.shape[1])
        token_time = flow_time[:, None].expand(-1, actions.shape[1])
        token_time = torch.where(
            prefix_temporal, torch.ones_like(token_time), token_time
        )
        noisy_actions = (
            token_time[..., None] * actions
            + (1.0 - token_time[..., None]) * noise
        ) * action_mask.to(actions.dtype)
        target_velocity = (actions - noise) * action_mask.to(actions.dtype)
        token_timesteps = (token_time * self.num_timestep_buckets).long()
        global_timestep = (flow_time * self.num_timestep_buckets).long()
        action_features = self._action_features(
            noisy_actions, token_timesteps, action_input.embodiment_id
        )
        suffix = torch.cat((conditioning.fixed_features, action_features), dim=1)
        suffix_mask = torch.cat(
            (conditioning.fixed_valid, action_mask.any(dim=-1)), dim=1
        )
        model_output, hidden_states = self._run_conditioned_model(
            suffix,
            suffix_mask,
            conditioning,
            global_timestep,
            return_all_hidden_states=True,
        )
        predicted_velocity = self.action_decoder(
            model_output, action_input.embodiment_id
        )[:, -actions.shape[1] :]
        loss_mask = postfix_loss_mask(action_mask, prefix_steps, self.contract)
        metrics = self._component_metrics(
            predicted_velocity, target_velocity, loss_mask
        )
        loss = float(self.config.control_flow_weight) * metrics["control_flow_mse"]
        if self.contract.has_delta_q:
            loss = loss + float(self.config.delta_q_flow_weight) * metrics[
                "delta_q_flow_mse"
            ]
        self._assert_finite_training_loss(loss)
        element_loss = (
            (predicted_velocity - target_velocity).square()
            * loss_mask.to(predicted_velocity.dtype)
        )
        return BatchFeature(
            data={
                "loss": loss,
                "action_loss": element_loss,
                "action_mask": loss_mask,
                "rtc_prefix_steps": prefix_steps,
                **metrics,
                "backbone_features": conditioning.backbone_features,
                "state_features": conditioning.state_features,
                "sensor_features": conditioning.sensor_features,
                "suffix_attention_mask": suffix_mask,
                "all_hidden_states": hidden_states,
            }
        )

    def _resolve_inference_prefix(
        self,
        action_input: BatchFeature,
        options: dict[str, Any] | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[int, torch.Tensor | None]:
        options = options or {}
        raw_prefix = options.get("rtc_prefix_action")
        prefix_steps = options.get("rtc_prefix_steps")
        if raw_prefix is not None:
            prefix = torch.as_tensor(raw_prefix, device=device, dtype=dtype)
            if prefix.ndim == 2:
                prefix = prefix.unsqueeze(0)
            if prefix_steps is None:
                prefix_steps = prefix.shape[1]
        elif prefix_steps:
            if "action" not in action_input:
                raise ValueError("RTC inference requires a normalized prefix action")
            prefix = action_input.action[:, : int(prefix_steps)].to(
                device=device, dtype=dtype
            )
        else:
            prefix = None
        prefix_steps = int(prefix_steps or 0)
        if not 0 <= prefix_steps <= self.config.rtc_training_max_prefix_steps:
            raise ValueError("RTC inference prefix exceeds checkpoint capability")
        if prefix_steps == 0:
            return 0, None
        if prefix is None:
            raise ValueError("nonzero RTC prefix steps require prefix actions")
        expected = (batch_size, prefix_steps, self.contract.expert_dim)
        if tuple(prefix.shape) != expected:
            raise ValueError(f"RTC prefix must have shape {expected}, got {tuple(prefix.shape)}")
        active = self.contract.active_mask(device=device).view(1, 1, -1)
        prefix = torch.where(active, prefix, torch.zeros_like(prefix))
        return prefix_steps, prefix

    @torch.no_grad()
    def _denoise(
        self,
        conditioning: BatchFeature,
        embodiment_id: torch.Tensor,
        action_input: BatchFeature,
        options: dict[str, Any] | None,
    ) -> BatchFeature:
        batch_size = conditioning.backbone_features.shape[0]
        active = self.contract.active_mask(
            device=conditioning.backbone_features.device
        ).view(1, 1, -1)
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            dtype=conditioning.backbone_features.dtype,
            device=conditioning.backbone_features.device,
        ) * active.to(conditioning.backbone_features.dtype)
        prefix_steps, prefix = self._resolve_inference_prefix(
            action_input,
            options,
            batch_size=batch_size,
            device=actions.device,
            dtype=actions.dtype,
        )
        if prefix is not None:
            actions = clamp_prefix(actions, prefix, self.contract)
        fixed_valid = conditioning.fixed_valid
        action_valid = torch.ones(
            batch_size,
            self.action_horizon,
            dtype=torch.bool,
            device=actions.device,
        )
        step_size = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            continuous_time = step / float(self.num_inference_timesteps)
            bucket = int(continuous_time * self.num_timestep_buckets)
            global_timestep = torch.full(
                (batch_size,), bucket, dtype=torch.long, device=actions.device
            )
            token_timesteps = global_timestep[:, None].expand(
                -1, self.action_horizon
            ).clone()
            if prefix_steps:
                token_timesteps[:, :prefix_steps] = self.num_timestep_buckets
                actions = clamp_prefix(actions, prefix, self.contract)
            action_features = self._action_features(
                actions, token_timesteps, embodiment_id
            )
            suffix = torch.cat((conditioning.fixed_features, action_features), dim=1)
            suffix_mask = torch.cat((fixed_valid, action_valid), dim=1)
            output = self._run_conditioned_model(
                suffix,
                suffix_mask,
                conditioning,
                global_timestep,
                return_all_hidden_states=False,
            )
            velocity = self.action_decoder(output, embodiment_id)[
                :, -self.action_horizon :
            ] * active.to(actions.dtype)
            if prefix_steps:
                velocity[:, :prefix_steps] = 0
            actions = (actions + step_size * velocity) * active.to(actions.dtype)
            if prefix_steps:
                actions = clamp_prefix(actions, prefix, self.contract)
        return BatchFeature(
            data={
                "action_pred": actions,
                "rtc_prefix_steps": torch.full(
                    (batch_size,),
                    prefix_steps,
                    dtype=torch.long,
                    device=actions.device,
                ),
                "backbone_features": conditioning.backbone_features,
                "state_features": conditioning.state_features,
                "sensor_features": conditioning.sensor_features,
            }
        )

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features,
        state_features,
        embodiment_id,
        backbone_output,
        action_input,
        options: dict[str, Any] | None = None,
        **conditioning_values,
    ):
        required = {
            "state_valid",
            "sensor_features",
            "sensor_valid",
            "fixed_features",
            "fixed_valid",
        }
        if not required.issubset(conditioning_values):
            conditioning = self._prepare_conditioning(backbone_output, action_input)
        else:
            conditioning = BatchFeature(
                data={
                    "backbone_output": backbone_output,
                    "backbone_features": backbone_features,
                    "state_features": state_features,
                    **conditioning_values,
                }
            )
        return self._denoise(conditioning, embodiment_id, action_input, options)

    @torch.no_grad()
    def get_action(self, backbone_output, action_input, options=None):
        conditioning = self._prepare_conditioning(backbone_output, action_input)
        return self._denoise(
            conditioning, action_input.embodiment_id, action_input, options
        )


__all__ = ["HacoActionHead"]
