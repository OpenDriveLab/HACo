"""Training-time real-time chunking adaptation of GR00T N1.7."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "GrootRTCConfig":
        from .config import GrootRTCConfig

        return GrootRTCConfig
    if name == "GrootRTC":
        from .model import GrootRTC

        return GrootRTC
    if name == "GrootRTCActionHead":
        from .action_head import GrootRTCActionHead

        return GrootRTCActionHead
    raise AttributeError(name)


__all__ = ["GrootRTC", "GrootRTCActionHead", "GrootRTCConfig"]
