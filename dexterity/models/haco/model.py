"""Standalone HACO model family rooted at official GR00T N1.7."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import tree
from torch import nn
from transformers import AutoConfig, AutoModel, AutoProcessor, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from dexterity.models.groot_n17.model import GrootN17, get_backbone_cls
from dexterity.models.groot_n17.processor import Gr00tN1d7DataCollator
from dexterity.models.pace.local_hf import resolve_local_model_path

from .action_head import HacoActionHead
from .config import HacoConfig
from .processor import HacoProcessor


HACO_EXTENSION_INIT_SEED = 42

_HACO_EXTENSION_KEY_PREFIXES = (
    "action_head.force_encoder.",
    "action_head.tactile_encoder.",
    "action_head.force_tactile_encoder.",
    "action_head.sensor_to_vlm.",
    "action_head.physical_type_embedding",
    "action_head.model.physical_cross_adapters.",
    "action_head.model.null_physical_token",
)


def _is_haco_extension_key(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in _HACO_EXTENSION_KEY_PREFIXES)


def _module_direct_parameter_keys(
    module_name: str, module: nn.Module
) -> tuple[str, ...]:
    prefix = f"{module_name}." if module_name else ""
    return tuple(
        f"{prefix}{name}" for name, _ in module.named_parameters(recurse=False)
    )


def _reset_builtin_module(module: nn.Module) -> bool:
    """Reset one parameter-owning leaf without recursing into children."""

    if isinstance(
        module,
        (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.Embedding,
        ),
    ):
        module.reset_parameters()
        return True
    if isinstance(module, nn.MultiheadAttention):
        # The out projection is a child Linear and is reset separately.
        module._reset_parameters()
        return True
    return False


def reinitialize_haco_extension_parameters(
    model: nn.Module,
    missing_keys: list[str] | tuple[str, ...] | set[str],
    *,
    seed: int = HACO_EXTENSION_INIT_SEED,
) -> dict[str, Any]:
    """Deterministically initialize only extensions absent from official GR00T.

    Transformers constructs models under a no-init context while loading large
    checkpoints. Parameters absent from the checkpoint can consequently retain
    uninitialized storage. This routine is driven by the exact missing-key set;
    it refuses non-HACO keys and partial resets of a parameter-owning leaf, so an
    official shared tensor cannot be overwritten.
    """

    requested = {str(key) for key in missing_keys}
    invalid = sorted(key for key in requested if not _is_haco_extension_key(key))
    if invalid:
        raise RuntimeError(
            "refusing to initialize non-HACO checkpoint keys: " f"{invalid}"
        )
    parameters = dict(model.named_parameters())
    unknown = sorted(requested.difference(parameters))
    if unknown:
        raise RuntimeError(
            "HACO missing keys are not model parameters: " f"{unknown}"
        )
    if not requested:
        return {
            "seed": int(seed),
            "reinitialized_keys": [],
            "all_finite": True,
        }
    meta = sorted(key for key in requested if parameters[key].is_meta)
    if meta:
        raise RuntimeError(
            "HACO extension parameters were not materialized by the loader: "
            f"{meta}"
        )

    special = {
        key
        for key in requested
        if key.endswith(".contextualizer.modality_embedding")
        or key.endswith(".contextualizer.hand_embedding.weight")
        or key.endswith(".contextualizer.digit_embedding.weight")
        or key == "action_head.physical_type_embedding"
        or key.endswith(".gate")
        or key.endswith(".null_physical_token")
    }
    handled: set[str] = set(special)
    modules = sorted(model.named_modules(), key=lambda item: item[0])
    cuda_devices = sorted(
        {
            int(parameters[key].device.index)
            for key in requested
            if parameters[key].is_cuda and parameters[key].device.index is not None
        }
    )
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        for module_name, module in modules:
            direct = set(_module_direct_parameter_keys(module_name, module))
            targets = (direct & requested) - special
            if not targets:
                continue
            if not direct.issubset(requested):
                shared = sorted(direct.difference(requested))
                raise RuntimeError(
                    "HACO extension reset would overwrite checkpoint tensors "
                    f"in {module_name!r}: {shared}"
                )
            if not _reset_builtin_module(module):
                raise RuntimeError(
                    "HACO extension parameter owner has no approved reset: "
                    f"{module_name} ({type(module).__name__})"
                )
            handled.update(targets)

        for key in sorted(special):
            parameter = parameters[key]
            if key.endswith(".contextualizer.modality_embedding"):
                nn.init.normal_(parameter, std=0.02)
            elif key.endswith(".contextualizer.hand_embedding.weight"):
                nn.init.normal_(parameter, std=0.02)
            elif key.endswith(".contextualizer.digit_embedding.weight"):
                nn.init.normal_(parameter, std=0.02)
            elif key == "action_head.physical_type_embedding":
                nn.init.normal_(
                    parameter,
                    std=float(model.config.physical_type_embedding_std),
                )
            elif key.endswith(".gate"):
                parameter.fill_(float(model.config.physical_cross_gate_init))
            else:
                parameter.zero_()

    unhandled = sorted(requested.difference(handled))
    if unhandled:
        raise RuntimeError(f"HACO extension keys were not initialized: {unhandled}")
    nonfinite = sorted(
        key for key in requested if not torch.isfinite(parameters[key]).all().item()
    )
    if nonfinite:
        raise RuntimeError(
            "HACO extension initialization produced non-finite parameters: "
            f"{nonfinite}"
        )
    gate_value = float(model.config.physical_cross_gate_init)
    if not math.isfinite(gate_value):
        raise RuntimeError("physical_cross_gate_init must be finite")
    return {
        "seed": int(seed),
        "reinitialized_keys": sorted(requested),
        "all_finite": True,
    }


class Haco(GrootN17):
    """The sole model class used by every HACO matrix row."""

    config_class = HacoConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: HacoConfig,
        transformers_loading_kwargs: dict[str, Any] | None = None,
    ) -> None:
        PreTrainedModel.__init__(self, config)
        self.config = config
        loading_kwargs = dict(
            transformers_loading_kwargs or {"trust_remote_code": True}
        )
        model_name = resolve_local_model_path(config.model_name, loading_kwargs)
        config.model_name = model_name
        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=loading_kwargs,
        )
        self.action_head = HacoActionHead(config)
        self.collator = Gr00tN1d7DataCollator(
            model_name=model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=loading_kwargs,
        )

    @classmethod
    def from_official_checkpoint(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: HacoConfig | None = None,
        output_loading_info: bool = False,
        **kwargs: Any,
    ):
        """Load official N1.7 weights and initialize only HACO extension keys.

        The source is rejected unless it is the official 40x132 GR00T family.
        This prevents accidental initialization from a task-posttrained HACO or
        GR00T-RTC checkpoint.
        """

        source = Path(pretrained_model_name_or_path).expanduser()
        config_path = source / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"official checkpoint config not found: {config_path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        expected = {
            "model_type": "Gr00tN1d7",
            "action_horizon": 40,
            "max_action_dim": 132,
        }
        mismatched = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"HACO requires the official GR00T N1.7 40x132 base: {mismatched}"
            )
        if config is None:
            config_overrides = kwargs.pop("config_overrides", {})
            config = HacoConfig.from_pretrained(
                str(source), **dict(config_overrides)
            )
        loaded = cls.from_pretrained(
            str(source),
            config=config,
            output_loading_info=True,
            **kwargs,
        )
        model, loading_info = loaded
        bad_missing = [
            key
            for key in loading_info.get("missing_keys", ())
            if not _is_haco_extension_key(key)
        ]
        bad_unexpected = list(loading_info.get("unexpected_keys", ()))
        mismatched_keys = list(loading_info.get("mismatched_keys", ()))
        if bad_missing or bad_unexpected or mismatched_keys:
            raise RuntimeError(
                "official GR00T -> HACO initialization was not boundary-compatible: "
                f"missing={bad_missing}, unexpected={bad_unexpected}, "
                f"mismatched={mismatched_keys}"
            )
        initialization = reinitialize_haco_extension_parameters(
            model,
            loading_info.get("missing_keys", ()),
        )
        loading_info["reinitialized_haco_extension_keys"] = initialization[
            "reinitialized_keys"
        ]
        loading_info["haco_extension_init_seed"] = initialization["seed"]
        loading_info["haco_extension_parameters_finite"] = initialization[
            "all_finite"
        ]
        model.config.official_base_checkpoint = str(source.resolve())
        if output_loading_info:
            return model, loading_info
        return model

    def prepare_input(self, inputs: dict) -> tuple[BatchFeature, BatchFeature]:
        inputs = dict(inputs)
        if "vlm_content" in inputs:
            contents = inputs.pop("vlm_content")
            if not isinstance(contents, list):
                contents = [contents]
            prepared = self.collator(
                [{"vlm_content": content} for content in contents]
            )["inputs"]
            inputs.update(prepared)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def move(value):
            if torch.is_floating_point(value):
                return value.to(self.device, dtype=self.dtype)
            return value.to(self.device)

        return (
            tree.map_structure(move, backbone_inputs),
            tree.map_structure(move, action_inputs),
        )

    def forward(self, inputs: dict) -> dict[str, Any]:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        return dict(self.action_head(self.backbone(backbone_inputs), action_inputs))

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None):
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        return self.action_head.get_action(
            self.backbone(backbone_inputs), action_inputs, options
        )

    @property
    def action_contract(self):
        return self.action_head.contract

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


AutoConfig.register(HacoConfig.model_type, HacoConfig, exist_ok=True)
AutoModel.register(HacoConfig, Haco, exist_ok=True)
AutoProcessor.register(HacoConfig.model_type, HacoProcessor, exist_ok=True)


__all__ = [
    "HACO_EXTENSION_INIT_SEED",
    "Haco",
    "reinitialize_haco_extension_parameters",
]
