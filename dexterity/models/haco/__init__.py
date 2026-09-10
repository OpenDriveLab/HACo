"""HACO: direct compliance actions, auxiliary delta-q, and trained RTC."""

from .config import HacoConfig
from .contract import (
    HACO_ACTION_CONTRACTS,
    HacoActionContract,
    get_action_contract,
)


def __getattr__(name: str):
    # Keep config/contract imports usable in light environments that do not
    # install the full GR00T runtime.
    if name == "Haco":
        from .model import Haco

        return Haco
    if name == "HacoActionHead":
        from .action_head import HacoActionHead

        return HacoActionHead
    if name == "HacoProcessor":
        from .processor import HacoProcessor

        return HacoProcessor
    raise AttributeError(name)


__all__ = [
    "HACO_ACTION_CONTRACTS",
    "Haco",
    "HacoActionContract",
    "HacoActionHead",
    "HacoConfig",
    "HacoProcessor",
    "get_action_contract",
]
