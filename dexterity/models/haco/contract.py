"""Action contracts for HACO.

HACO deliberately keeps the official GR00T 132-D expert carrier so that
the official N1.7 action encoder/decoder weights remain loadable.  The carrier
layout is a serialization detail; only the dimensions selected by
``active_mask`` belong to an experiment's action vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch


HACO_ACTION_HORIZON: Final = 40
HACO_EXPERT_ACTION_DIM: Final = 132
HACO_WRIST_DIM: Final = 18
HACO_JOINT_DIM: Final = 44

ACTION_CONTRACT_NAMES: Final = (
    "joint_compliance_delta",
    "compliance_only",
    "nominal_only",
)


@dataclass(frozen=True)
class HacoActionContract:
    """One immutable HACO action vocabulary inside the 132-D carrier."""

    name: str
    q_semantics: str
    has_delta_q: bool
    horizon: int = HACO_ACTION_HORIZON
    wrist_dim: int = HACO_WRIST_DIM
    joint_dim: int = HACO_JOINT_DIM
    expert_dim: int = HACO_EXPERT_ACTION_DIM

    def __post_init__(self) -> None:
        if self.name not in ACTION_CONTRACT_NAMES:
            raise ValueError(f"unknown HACO action contract {self.name!r}")
        if self.q_semantics not in ("q_compliance", "q_nominal"):
            raise ValueError(f"invalid q semantics {self.q_semantics!r}")
        if self.has_delta_q != (self.name == "joint_compliance_delta"):
            raise ValueError("delta-q is exclusive to joint_compliance_delta")
        if self.has_delta_q and self.q_semantics != "q_compliance":
            raise ValueError("joint delta-q supervision requires q_compliance")

    @property
    def action_target(self) -> str:
        return self.q_semantics

    @property
    def semantic_dim(self) -> int:
        return self.wrist_dim + self.joint_dim * (2 if self.has_delta_q else 1)

    @property
    def padding_dim(self) -> int:
        return self.expert_dim - self.semantic_dim

    @property
    def wrist_slice(self) -> slice:
        return slice(0, self.wrist_dim)

    @property
    def q_slice(self) -> slice:
        return slice(self.wrist_dim, self.wrist_dim + self.joint_dim)

    @property
    def delta_q_slice(self) -> slice | None:
        if not self.has_delta_q:
            return None
        return slice(
            self.wrist_dim + self.joint_dim,
            self.wrist_dim + 2 * self.joint_dim,
        )

    @property
    def padding_slice(self) -> slice:
        return slice(self.semantic_dim, self.expert_dim)

    @property
    def component_names(self) -> tuple[str, ...]:
        components = ("wrist", self.q_semantics)
        return components + (("delta_q",) if self.has_delta_q else ())

    def active_mask(self, *, device=None) -> torch.Tensor:
        mask = torch.zeros(self.expert_dim, dtype=torch.bool, device=device)
        mask[: self.semantic_dim] = True
        return mask

    def _validate_component(
        self,
        value: torch.Tensor,
        *,
        name: str,
        prefix: tuple[int, ...],
        width: int,
    ) -> None:
        expected = (*prefix, width)
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")

    def pack(
        self,
        wrist: torch.Tensor,
        *,
        q_compliance: torch.Tensor | None = None,
        q_nominal: torch.Tensor | None = None,
        delta_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pack only the components owned by this contract.

        In particular, ``compliance_only`` and ``nominal_only`` reject a
        delta-q tensor rather than silently masking it. This keeps delta-free
        data pipelines free of delta-q inputs and statistics.
        """

        prefix = tuple(wrist.shape[:-1])
        self._validate_component(
            wrist, name="wrist", prefix=prefix, width=self.wrist_dim
        )
        q_by_name = {
            "q_compliance": q_compliance,
            "q_nominal": q_nominal,
        }
        q = q_by_name[self.q_semantics]
        wrong_q = q_nominal if self.q_semantics == "q_compliance" else q_compliance
        if q is None:
            raise ValueError(f"{self.name} requires {self.q_semantics}")
        if wrong_q is not None:
            raise ValueError(f"{self.name} does not accept the other q target")
        self._validate_component(
            q, name=self.q_semantics, prefix=prefix, width=self.joint_dim
        )
        components = [wrist, q]
        if self.has_delta_q:
            if delta_q is None:
                raise ValueError("joint_compliance_delta requires delta_q")
            self._validate_component(
                delta_q, name="delta_q", prefix=prefix, width=self.joint_dim
            )
            components.append(delta_q)
        elif delta_q is not None:
            raise ValueError(f"{self.name} must not receive delta_q")
        padding = wrist.new_zeros(*prefix, self.padding_dim)
        return torch.cat((*components, padding), dim=-1)

    def validate_carrier(self, action: torch.Tensor) -> None:
        if action.shape[-1] != self.expert_dim:
            raise ValueError(
                f"{self.name} expects [...,{self.expert_dim}], got {tuple(action.shape)}"
            )

    def unpack(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        self.validate_carrier(action)
        result = {
            "wrist": action[..., self.wrist_slice],
            self.q_semantics: action[..., self.q_slice],
            "padding": action[..., self.padding_slice],
        }
        if self.has_delta_q:
            result["delta_q"] = action[..., self.delta_q_slice]
        return result

    def executable(self, action: torch.Tensor) -> torch.Tensor:
        """Return wrist + q exactly as predicted, never ``q + delta_q``."""

        self.validate_carrier(action)
        return action[..., : self.wrist_dim + self.joint_dim]

    def executable_q(self, action: torch.Tensor) -> torch.Tensor:
        """Return the directly executable q target for robot-hand control."""

        self.validate_carrier(action)
        return action[..., self.q_slice]


HACO_ACTION_CONTRACTS: Final = {
    "joint_compliance_delta": HacoActionContract(
        name="joint_compliance_delta",
        q_semantics="q_compliance",
        has_delta_q=True,
    ),
    "compliance_only": HacoActionContract(
        name="compliance_only",
        q_semantics="q_compliance",
        has_delta_q=False,
    ),
    "nominal_only": HacoActionContract(
        name="nominal_only",
        q_semantics="q_nominal",
        has_delta_q=False,
    ),
}


def get_action_contract(name: str) -> HacoActionContract:
    try:
        return HACO_ACTION_CONTRACTS[str(name)]
    except KeyError as error:
        raise ValueError(
            f"action_contract must be one of {ACTION_CONTRACT_NAMES}, got {name!r}"
        ) from error


__all__ = [
    "ACTION_CONTRACT_NAMES",
    "HACO_ACTION_CONTRACTS",
    "HACO_ACTION_HORIZON",
    "HACO_EXPERT_ACTION_DIM",
    "HacoActionContract",
    "get_action_contract",
]
