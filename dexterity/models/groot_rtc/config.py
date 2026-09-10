"""Configuration for the trained-RTC GR00T N1.7 derivative."""

from __future__ import annotations

import math
from typing import Any

from dexterity.models.groot_n17.config import GrootN17Config


RTC_CHECKPOINT_SCHEMA = "dreamzero.groot_rtc_checkpoint.v1"
RTC_MODEL_TYPE = "Gr00tN1d7RTC"


def default_prefix_weights(max_prefix_steps: int, preferred_steps: int) -> list[float]:
    """Return uniform categorical weights with a two-times preferred bucket."""

    weights = [1.0] * (int(max_prefix_steps) + 1)
    if 0 <= int(preferred_steps) <= int(max_prefix_steps):
        weights[int(preferred_steps)] = 2.0
    return weights


class GrootRTCConfig(GrootN17Config):
    """GR00T N1.7 config with explicit trained-RTC checkpoint metadata."""

    model_type = RTC_MODEL_TYPE

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("model_type", None)
        kwargs.pop("architectures", None)
        max_prefix_steps = int(kwargs.pop("rtc_training_max_prefix_steps", 12))
        inference_prefix_steps = int(kwargs.pop("rtc_inference_prefix_steps", 10))
        raw_weights = kwargs.pop("rtc_training_prefix_weights", None)
        weights = (
            default_prefix_weights(max_prefix_steps, inference_prefix_steps)
            if raw_weights is None
            else [float(value) for value in raw_weights]
        )
        rtc_schema = str(kwargs.pop("rtc_schema", RTC_CHECKPOINT_SCHEMA))
        rtc_sampling = str(kwargs.pop("rtc_training_sampling", "categorical"))
        rtc_action_frame = str(kwargs.pop("rtc_action_frame", "absolute"))

        # Trained RTC is intentionally a separate absolute-action model family.
        kwargs["use_relative_action"] = False
        super().__init__(**kwargs)
        self.model_type = type(self).model_type
        self.architectures = ["GrootRTC"]
        self.rtc_schema = rtc_schema
        self.rtc_training_max_prefix_steps = max_prefix_steps
        self.rtc_training_prefix_weights = weights
        self.rtc_training_sampling = rtc_sampling
        self.rtc_inference_prefix_steps = inference_prefix_steps
        self.rtc_action_frame = rtc_action_frame
        self._validate_rtc()

    def _validate_rtc(self) -> None:
        if self.rtc_schema != RTC_CHECKPOINT_SCHEMA:
            raise ValueError(
                f"rtc_schema must be {RTC_CHECKPOINT_SCHEMA!r}, got {self.rtc_schema!r}"
            )
        if self.rtc_training_sampling != "categorical":
            raise ValueError("rtc_training_sampling must be 'categorical'")
        if self.rtc_action_frame != "absolute":
            raise ValueError("GrootRTC supports absolute actions only")
        if not 0 <= self.rtc_training_max_prefix_steps < int(self.action_horizon):
            raise ValueError(
                "rtc_training_max_prefix_steps must be in "
                f"[0, action_horizon), got {self.rtc_training_max_prefix_steps} "
                f"for horizon {self.action_horizon}"
            )
        if not (
            0
            <= self.rtc_inference_prefix_steps
            <= self.rtc_training_max_prefix_steps
        ):
            raise ValueError(
                "rtc_inference_prefix_steps must be supported by the trained "
                "prefix range"
            )
        expected = self.rtc_training_max_prefix_steps + 1
        if len(self.rtc_training_prefix_weights) != expected:
            raise ValueError(
                "rtc_training_prefix_weights must contain one value for every prefix "
                f"length 0..{self.rtc_training_max_prefix_steps}"
            )
        if any(
            not math.isfinite(value) or value < 0
            for value in self.rtc_training_prefix_weights
        ):
            raise ValueError(
                "rtc_training_prefix_weights must be finite and nonnegative"
            )
        if sum(self.rtc_training_prefix_weights) <= 0:
            raise ValueError(
                "rtc_training_prefix_weights must contain a positive value"
            )

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict[str, Any]:
        config = super().to_filtered_dict(exclude_augment=exclude_augment)
        config.update(
            {
                "model_type": self.model_type,
                "architectures": list(self.architectures),
                "rtc_schema": self.rtc_schema,
                "rtc_training_max_prefix_steps": self.rtc_training_max_prefix_steps,
                "rtc_training_prefix_weights": list(self.rtc_training_prefix_weights),
                "rtc_training_sampling": self.rtc_training_sampling,
                "rtc_inference_prefix_steps": self.rtc_inference_prefix_steps,
                "rtc_action_frame": self.rtc_action_frame,
                "use_relative_action": False,
            }
        )
        return config


__all__ = [
    "GrootRTCConfig",
    "RTC_CHECKPOINT_SCHEMA",
    "RTC_MODEL_TYPE",
    "default_prefix_weights",
]
