from __future__ import annotations

import torch
from torch import nn

from .sensor_common import FingerTokenContextualizer, make_mlp


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(8, output_channels)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, 3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(8, output_channels)
        self.activation = nn.GELU()
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(8, output_channels),
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        value = self.activation(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return self.activation(value + residual)


class DeformationEncoder(nn.Module):
    """Encode native 240x240 deformation maps into one token per finger."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
            nn.Conv2d(256, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projection = make_mlp(
            64 * 4 * 4,
            max(512, output_dim * 2),
            output_dim,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(images).flatten(start_dim=1))


class TactileFingerEncoder(nn.Module):
    """Encode wrench history and current deformation into ten tactile tokens."""

    def __init__(self, config) -> None:
        super().__init__()
        self.history_length = int(config.tactile_history_length)
        self.finger_count = int(config.tactile_finger_count)
        self.wrench_dim = int(config.tactile_wrench_dim)
        self.image_size = int(config.tactile_image_size)
        self.encoder_dim = int(config.sensor_encoder_dim)
        wrench_input_dim = self.history_length * self.wrench_dim + self.history_length
        self.wrench_mlp = make_mlp(wrench_input_dim, self.encoder_dim, self.encoder_dim)
        self.deformation_encoder = DeformationEncoder(self.encoder_dim)
        self.fusion_mlp = make_mlp(
            self.encoder_dim * 2 + 2, self.encoder_dim * 2, self.encoder_dim
        )

    def forward(
        self,
        wrench_history: torch.Tensor,
        wrench_valid: torch.Tensor,
        deformation: torch.Tensor,
        deformation_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_wrench = (self.finger_count, self.history_length, self.wrench_dim)
        if tuple(wrench_history.shape[-3:]) != expected_wrench:
            raise ValueError(
                f"tactile_wrench_history must end in {expected_wrench}, got "
                f"{tuple(wrench_history.shape)}"
            )
        if wrench_valid.shape != wrench_history.shape[:-1]:
            raise ValueError("tactile_wrench_valid shape does not match wrench history")
        batch_size = wrench_history.shape[0]
        expected_image = (batch_size, self.finger_count, self.image_size, self.image_size)
        if tuple(deformation.shape) != expected_image:
            raise ValueError(
                f"tactile_deformation must have shape {expected_image}, got "
                f"{tuple(deformation.shape)}"
            )
        if tuple(deformation_valid.shape) != expected_image[:2]:
            raise ValueError("tactile_deformation_valid shape does not match deformation")

        wrench_valid = wrench_valid.bool() & torch.isfinite(wrench_history).all(dim=-1)
        wrench = torch.where(
            wrench_valid.unsqueeze(-1), wrench_history, torch.zeros_like(wrench_history)
        )
        wrench_input = torch.cat(
            (wrench.flatten(start_dim=-2), wrench_valid.to(wrench.dtype)), dim=-1
        )
        wrench_feature = self.wrench_mlp(wrench_input)
        wrench_available = wrench_valid.any(dim=-1)
        wrench_feature = wrench_feature * wrench_available.unsqueeze(-1).to(
            wrench_feature.dtype
        )

        images = deformation.to(dtype=wrench_history.dtype)
        if deformation.dtype == torch.uint8:
            images = images / 255.0
        image_available = (
            deformation_valid.bool() & torch.isfinite(images).flatten(-2).all(dim=-1)
        )
        images = torch.where(
            image_available[..., None, None], images, torch.zeros_like(images)
        )
        image_feature = self.deformation_encoder(
            images.reshape(-1, 1, self.image_size, self.image_size)
        ).reshape(batch_size, self.finger_count, self.encoder_dim)
        image_feature = image_feature * image_available.unsqueeze(-1).to(
            image_feature.dtype
        )

        fusion_input = torch.cat(
            (
                wrench_feature,
                image_feature,
                wrench_available.unsqueeze(-1).to(wrench_feature.dtype),
                image_available.unsqueeze(-1).to(wrench_feature.dtype),
            ),
            dim=-1,
        )
        local_tokens = self.fusion_mlp(fusion_input)
        finger_valid = wrench_available | image_available
        local_tokens = local_tokens * finger_valid.unsqueeze(-1).to(local_tokens.dtype)
        return local_tokens, finger_valid


class TactileEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.local_encoder = TactileFingerEncoder(config)
        self.contextualizer = FingerTokenContextualizer(
            config, add_modality_embedding=True
        )

    def encode_local(self, wrench_history, wrench_valid, deformation, deformation_valid):
        return self.local_encoder(
            wrench_history, wrench_valid, deformation, deformation_valid
        )

    def forward(self, wrench_history, wrench_valid, deformation, deformation_valid):
        return self.contextualizer(
            *self.encode_local(
                wrench_history, wrench_valid, deformation, deformation_valid
            )
        )
