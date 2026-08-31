"""Lightweight UV completion student with explicit fusion ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - model construction requires torch.
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover - keep package import-safe.
    torch = None
    F = None
    nn = None


UV_COMPLETION_MODES = {
    "no_geometry",
    "naive_concat",
    "film_visibility",
    "full_router",
}

UNCERTAINTY_PARAMETERIZATIONS = {
    "clamped_log_variance",
    "softplus_variance",
}


@dataclass(frozen=True)
class UVCompletionLiteConfig:
    mode: str = "full_router"
    input_size: int = 256
    base_channels: int = 24
    geometry_channels: int = 6
    normalization: str = "batch"
    uncertainty_parameterization: str = "clamped_log_variance"
    max_parameters: int = 750_000


@dataclass
class UVCompletionOutputs:
    completed_uv: Any
    rgb_residual: Any
    log_variance: Any
    confidence: Any
    mode: str


def normalization_layer(channels: int, normalization: str):
    if nn is None:
        raise RuntimeError("normalization_layer requires torch")
    if normalization == "batch":
        return nn.BatchNorm2d(channels)
    if normalization == "group":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unsupported_uv_completion_normalization:{normalization}")


class DepthwiseSeparableBlock(nn.Module if nn is not None else object):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        normalization: str = "batch",
    ) -> None:
        if nn is None:
            raise RuntimeError("DepthwiseSeparableBlock requires torch")
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            normalization_layer(in_channels, normalization),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            normalization_layer(out_channels, normalization),
            nn.SiLU(inplace=True),
        )

    def forward(self, value):  # type: ignore[override]
        return self.block(value)


class ResidualDepthwiseBlock(nn.Module if nn is not None else object):
    def __init__(self, channels: int, normalization: str = "batch") -> None:
        if nn is None:
            raise RuntimeError("ResidualDepthwiseBlock requires torch")
        super().__init__()
        self.block = DepthwiseSeparableBlock(
            channels,
            channels,
            normalization=normalization,
        )

    def forward(self, value):  # type: ignore[override]
        return value + self.block(value)


class GeometryConditioner(nn.Module if nn is not None else object):
    def __init__(
        self,
        geometry_channels: int,
        feature_channels: int,
        normalization: str = "batch",
    ) -> None:
        if nn is None:
            raise RuntimeError("GeometryConditioner requires torch")
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(geometry_channels + 1, feature_channels, kernel_size=1, bias=False),
            normalization_layer(feature_channels, normalization),
            nn.SiLU(inplace=True),
        )
        self.film = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels, feature_channels * 2),
        )

    def forward(self, geometry, visibility, target_size: tuple[int, int]):  # type: ignore[override]
        spatial = self.spatial(torch.cat([geometry, visibility], dim=1))
        spatial = F.interpolate(spatial, size=target_size, mode="bilinear", align_corners=False)
        gamma, beta = self.film(spatial).chunk(2, dim=1)
        return spatial, gamma[:, :, None, None], beta[:, :, None, None]


class UVCompletionLite(nn.Module if nn is not None else object):
    """Complete missing UV regions while preserving visible pixels."""

    def __init__(self, config: UVCompletionLiteConfig | None = None) -> None:
        if torch is None or nn is None or F is None:
            raise RuntimeError("UVCompletionLite requires torch")
        super().__init__()
        self.config = config or UVCompletionLiteConfig()
        if self.config.mode not in UV_COMPLETION_MODES:
            raise ValueError(f"unsupported_uv_completion_mode:{self.config.mode}")
        if self.config.base_channels < 8:
            raise ValueError("uv_completion_base_channels_must_be_at_least_8")
        if self.config.geometry_channels < 1:
            raise ValueError("uv_completion_geometry_channels_must_be_positive")
        if self.config.normalization not in {"batch", "group"}:
            raise ValueError(
                f"unsupported_uv_completion_normalization:{self.config.normalization}"
            )
        if self.config.uncertainty_parameterization not in UNCERTAINTY_PARAMETERIZATIONS:
            raise ValueError(
                "unsupported_uv_completion_uncertainty_parameterization:"
                f"{self.config.uncertainty_parameterization}"
            )

        base = self.config.base_channels
        normalization = self.config.normalization
        input_channels = 4 + (self.config.geometry_channels if self.config.mode == "naive_concat" else 0)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base, kernel_size=3, padding=1, bias=False),
            normalization_layer(base, normalization),
            nn.SiLU(inplace=True),
        )
        self.encoder_1 = ResidualDepthwiseBlock(base, normalization)
        self.down_1 = DepthwiseSeparableBlock(
            base,
            base * 2,
            stride=2,
            normalization=normalization,
        )
        self.encoder_2 = ResidualDepthwiseBlock(base * 2, normalization)
        self.down_2 = DepthwiseSeparableBlock(
            base * 2,
            base * 4,
            stride=2,
            normalization=normalization,
        )
        self.bottleneck = nn.Sequential(
            ResidualDepthwiseBlock(base * 4, normalization),
            ResidualDepthwiseBlock(base * 4, normalization),
        )
        self.geometry_conditioner = (
            GeometryConditioner(
                self.config.geometry_channels,
                base * 4,
                normalization,
            )
            if self.config.mode in {"film_visibility", "full_router"}
            else None
        )
        self.decoder_1 = nn.Sequential(
            DepthwiseSeparableBlock(
                base * 6,
                base * 2,
                normalization=normalization,
            ),
            ResidualDepthwiseBlock(base * 2, normalization),
        )
        self.decoder_2 = nn.Sequential(
            DepthwiseSeparableBlock(
                base * 3,
                base,
                normalization=normalization,
            ),
            ResidualDepthwiseBlock(base, normalization),
        )
        self.output_head = nn.Conv2d(base, 4, kernel_size=1)

        parameter_count = count_uv_completion_parameters(self)
        if parameter_count > self.config.max_parameters:
            raise ValueError(
                f"uv_completion_parameter_budget_exceeded:{parameter_count}>{self.config.max_parameters}"
            )

    def _validate_inputs(self, partial_uv, visibility, geometry) -> None:
        if partial_uv.ndim != 4 or partial_uv.shape[1] != 3:
            raise ValueError("partial_uv_shape_must_be_bx3xhxw")
        if visibility.ndim != 4 or visibility.shape[1] != 1:
            raise ValueError("visibility_shape_must_be_bx1xhxw")
        if partial_uv.shape[0] != visibility.shape[0] or partial_uv.shape[2:] != visibility.shape[2:]:
            raise ValueError("partial_uv_visibility_shape_mismatch")
        if partial_uv.shape[2] % 4 or partial_uv.shape[3] % 4:
            raise ValueError("uv_spatial_dimensions_must_be_divisible_by_4")
        if not bool(torch.isfinite(partial_uv).all()) or not bool(torch.isfinite(visibility).all()):
            raise ValueError("uv_inputs_contain_nan_or_inf")
        if bool((visibility < 0).any()) or bool((visibility > 1).any()):
            raise ValueError("visibility_values_must_be_between_0_and_1")
        geometry_required = self.config.mode != "no_geometry"
        if geometry_required and geometry is None:
            raise ValueError(f"geometry_required_for_mode:{self.config.mode}")
        if geometry is not None:
            expected = (
                partial_uv.shape[0],
                self.config.geometry_channels,
                partial_uv.shape[2],
                partial_uv.shape[3],
            )
            if tuple(geometry.shape) != expected:
                raise ValueError(f"geometry_shape_mismatch:{tuple(geometry.shape)}!={expected}")
            if not bool(torch.isfinite(geometry).all()):
                raise ValueError("geometry_contains_nan_or_inf")

    def forward(self, partial_uv, visibility, geometry=None):  # type: ignore[override]
        self._validate_inputs(partial_uv, visibility, geometry)
        inputs = [partial_uv, visibility]
        if self.config.mode == "naive_concat":
            inputs.append(geometry)
        skip_1 = self.encoder_1(self.stem(torch.cat(inputs, dim=1)))
        skip_2 = self.encoder_2(self.down_1(skip_1))
        bottleneck = self.bottleneck(self.down_2(skip_2))

        if self.geometry_conditioner is not None:
            spatial, gamma, beta = self.geometry_conditioner(
                geometry,
                visibility,
                target_size=(bottleneck.shape[2], bottleneck.shape[3]),
            )
            bottleneck = (bottleneck + spatial) * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta

        up_1 = F.interpolate(bottleneck, size=skip_2.shape[2:], mode="bilinear", align_corners=False)
        up_1 = self.decoder_1(torch.cat([up_1, skip_2], dim=1))
        up_2 = F.interpolate(up_1, size=skip_1.shape[2:], mode="bilinear", align_corners=False)
        features = self.decoder_2(torch.cat([up_2, skip_1], dim=1))
        raw_output = self.output_head(features)
        predicted_uv = torch.sigmoid(raw_output[:, :3])
        rgb_residual = predicted_uv - partial_uv
        if self.config.uncertainty_parameterization == "softplus_variance":
            variance = 1e-6 + F.softplus(raw_output[:, 3:4])
            log_variance = torch.log(variance)
        else:
            log_variance = raw_output[:, 3:4].clamp(-6.0, 3.0)
        confidence = torch.sigmoid(-log_variance)

        if self.config.mode == "full_router":
            fallback_uv = torch.flip(partial_uv, dims=[3])
            missing_uv = confidence * predicted_uv + (1.0 - confidence) * fallback_uv
        else:
            missing_uv = predicted_uv
        completed_uv = visibility * partial_uv + (1.0 - visibility) * missing_uv
        return UVCompletionOutputs(
            completed_uv=completed_uv,
            rgb_residual=rgb_residual,
            log_variance=log_variance,
            confidence=confidence,
            mode=self.config.mode,
        )


def count_uv_completion_parameters(model: Any, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (value for value in parameters if value.requires_grad)
    return int(sum(value.numel() for value in parameters))
