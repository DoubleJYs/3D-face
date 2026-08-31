"""Lightweight complete interface for the parent structure-guided UV work.

The module keeps the parent paper's distinct mechanisms explicit: a texture
branch, a structure branch, structure-guidance gating (SGG), local
cross-attention (CA), enhanced texture context (ETC), uncertainty, and a
training-only three-branch discriminator.  It is an implementation of the
mechanism contract, not a recovered copy of unavailable original weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

try:  # pragma: no cover - construction requires torch.
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover - keep package import-safe.
    torch = None
    F = None
    nn = None


@dataclass(frozen=True)
class StructureGuidedUVConfig:
    input_size: int = 256
    structure_channels: int = 6
    expression_token_dim: int = 32
    base_channels: int = 12
    dynamic_rank: int = 4
    attention_window: int = 3
    maximum_parameters: int = 250_000


@dataclass
class StructureGuidedUVOutputs:
    completed_uv: Any
    predicted_uv: Any
    rgb_residual: Any
    log_variance: Any
    confidence: Any
    structure_prediction: Any
    structure_gate: Any
    cross_attention: Any
    expression_coefficients: Any
    enhanced_features: Any


@dataclass
class MultiBranchDiscriminatorOutputs:
    global_texture: Any
    structural_boundary: Any
    masked_patch: Any


def _group_norm(channels: int):
    if nn is None:
        raise RuntimeError("structure_guided_uv_requires_torch")
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class GatedConvolution(nn.Module if nn is not None else object):
    """Feature and learned gate convolution used by both parent branches."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        if nn is None:
            raise RuntimeError("structure_guided_uv_requires_torch")
        super().__init__()
        self.feature = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.gate = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.normalization = _group_norm(out_channels)

    def forward(self, value: Any) -> Any:  # type: ignore[override]
        return F.silu(self.normalization(self.feature(value))) * torch.sigmoid(self.gate(value))


class StructureGuidanceGate(nn.Module if nn is not None else object):
    """SGG: visibility-aware gate over the structure representation."""

    def __init__(self, channels: int) -> None:
        if nn is None:
            raise RuntimeError("structure_guided_uv_requires_torch")
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, structure: Any, visibility: Any) -> tuple[Any, Any]:  # type: ignore[override]
        resized = F.interpolate(visibility, size=structure.shape[2:], mode="nearest")
        gate = self.gate(torch.cat((structure, resized), dim=1))
        return structure * gate, gate


class LocalCrossAttention(nn.Module if nn is not None else object):
    """Linear-cost local texture-query/structure-key-value attention."""

    def __init__(self, channels: int, window: int) -> None:
        if nn is None:
            raise RuntimeError("structure_guided_uv_requires_torch")
        super().__init__()
        if window < 1 or window % 2 == 0:
            raise ValueError("cross_attention_window_must_be_positive_odd")
        self.channels = channels
        self.window = window
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, texture: Any, structure: Any) -> tuple[Any, Any]:  # type: ignore[override]
        batch, channels, height, width = texture.shape
        query = self.query(texture).reshape(batch, channels, 1, height * width)
        padding = self.window // 2
        key = F.unfold(self.key(structure), self.window, padding=padding)
        value = F.unfold(self.value(structure), self.window, padding=padding)
        neighbors = self.window * self.window
        key = key.reshape(batch, channels, neighbors, height * width)
        value = value.reshape(batch, channels, neighbors, height * width)
        scores = (query * key).sum(dim=1) / sqrt(float(channels))
        attention = torch.softmax(scores, dim=1)
        context = (attention[:, None] * value).sum(dim=2).reshape(
            batch, channels, height, width
        )
        return self.output(context), attention.reshape(batch, neighbors, height, width)


class ExpressionDynamicModulation(nn.Module if nn is not None else object):
    """Continuous expression-conditioned low-rank parameters for B."""

    def __init__(self, token_dim: int, channels: int, rank: int) -> None:
        if nn is None:
            raise RuntimeError("structure_guided_uv_requires_torch")
        super().__init__()
        self.coefficients = nn.Linear(token_dim, rank)
        self.basis = nn.Parameter(torch.zeros(rank, channels * 2))
        nn.init.normal_(self.basis, mean=0.0, std=0.01)

    def forward(self, value: Any, token: Any) -> tuple[Any, Any]:  # type: ignore[override]
        coefficients = torch.softmax(self.coefficients(token), dim=1)
        gamma, beta = (coefficients @ self.basis).chunk(2, dim=1)
        modulated = value * (1.0 + 0.25 * torch.tanh(gamma)[:, :, None, None])
        modulated = modulated + 0.25 * torch.tanh(beta)[:, :, None, None]
        return modulated, coefficients


class StructureGuidedUVCompletion(nn.Module if nn is not None else object):
    """SGG-CA-ETC UV completion with exact visible-pixel preservation."""

    def __init__(self, config: StructureGuidedUVConfig | None = None) -> None:
        if torch is None or nn is None or F is None:
            raise RuntimeError("structure_guided_uv_requires_torch")
        super().__init__()
        self.config = config or StructureGuidedUVConfig()
        cfg = self.config
        if cfg.base_channels < 8 or cfg.structure_channels < 1:
            raise ValueError("structure_guided_uv_channel_contract_invalid")
        if not 2 <= cfg.dynamic_rank <= 16:
            raise ValueError("structure_guided_uv_dynamic_rank_invalid")
        base = cfg.base_channels
        self.texture_stem = GatedConvolution(4, base)
        self.texture_down_1 = GatedConvolution(base, base * 2, stride=2)
        self.texture_down_2 = GatedConvolution(base * 2, base * 4, stride=2)
        self.structure_stem = GatedConvolution(cfg.structure_channels + 1, base)
        self.structure_down_1 = GatedConvolution(base, base * 2, stride=2)
        self.structure_down_2 = GatedConvolution(base * 2, base * 4, stride=2)
        self.sgg = StructureGuidanceGate(base * 4)
        self.cross_attention = LocalCrossAttention(base * 4, cfg.attention_window)
        self.expression_modulation = ExpressionDynamicModulation(
            cfg.expression_token_dim, base * 4, cfg.dynamic_rank
        )
        self.etc = nn.Sequential(
            nn.Conv2d(base * 12, base * 4, 1, bias=False),
            _group_norm(base * 4),
            nn.SiLU(inplace=True),
            GatedConvolution(base * 4, base * 4),
        )
        self.texture_up_1 = GatedConvolution(base * 6, base * 2)
        self.texture_up_2 = GatedConvolution(base * 3, base)
        self.texture_head = nn.Conv2d(base, 4, 1)
        # The complete parent-B mechanism starts from the frozen B-lite
        # completion during restoration diagnostics.  A zero head therefore
        # means an exact, auditable no-change initialization instead of a
        # random UV replacement.
        nn.init.zeros_(self.texture_head.weight)
        nn.init.zeros_(self.texture_head.bias)
        self.structure_up_1 = GatedConvolution(base * 6, base * 2)
        self.structure_up_2 = GatedConvolution(base * 3, base)
        self.structure_head = nn.Conv2d(base, 1, 1)
        count = count_structure_guided_uv_parameters(self)
        if count > cfg.maximum_parameters:
            raise ValueError(
                f"structure_guided_uv_parameter_budget_exceeded:{count}>{cfg.maximum_parameters}"
            )

    def _validate(
        self,
        partial_uv: Any,
        visibility: Any,
        structure: Any,
        expression: Any,
        base_completion: Any | None,
    ) -> None:
        if partial_uv.ndim != 4 or partial_uv.shape[1] != 3:
            raise ValueError("partial_uv_shape_must_be_bx3xhxw")
        mask_shape = (partial_uv.shape[0], 1, partial_uv.shape[2], partial_uv.shape[3])
        structure_shape = (
            partial_uv.shape[0],
            self.config.structure_channels,
            partial_uv.shape[2],
            partial_uv.shape[3],
        )
        if tuple(visibility.shape) != mask_shape or tuple(structure.shape) != structure_shape:
            raise ValueError("structure_guided_uv_spatial_contract_invalid")
        if tuple(expression.shape) != (
            partial_uv.shape[0], self.config.expression_token_dim
        ):
            raise ValueError("structure_guided_uv_expression_contract_invalid")
        if partial_uv.shape[2] % 4 or partial_uv.shape[3] % 4:
            raise ValueError("structure_guided_uv_size_must_be_divisible_by_four")
        values = (partial_uv, visibility, structure, expression)
        if base_completion is not None:
            if tuple(base_completion.shape) != tuple(partial_uv.shape):
                raise ValueError("structure_guided_uv_base_completion_shape_invalid")
            values = (*values, base_completion)
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("structure_guided_uv_nonfinite_input")
        if bool((visibility < 0).any()) or bool((visibility > 1).any()):
            raise ValueError("structure_guided_uv_visibility_out_of_range")

    def forward(  # type: ignore[override]
        self,
        partial_uv: Any,
        visibility: Any,
        structure: Any,
        expression_token: Any,
        base_completion: Any | None = None,
    ) -> StructureGuidedUVOutputs:
        self._validate(
            partial_uv, visibility, structure, expression_token, base_completion
        )
        texture_0 = self.texture_stem(torch.cat((partial_uv, visibility), dim=1))
        texture_1 = self.texture_down_1(texture_0)
        texture_2 = self.texture_down_2(texture_1)
        structure_0 = self.structure_stem(torch.cat((structure, visibility), dim=1))
        structure_1 = self.structure_down_1(structure_0)
        structure_2 = self.structure_down_2(structure_1)
        guided, gate = self.sgg(structure_2, visibility)
        attended, attention = self.cross_attention(texture_2, guided)
        modulated, coefficients = self.expression_modulation(texture_2, expression_token)
        enhanced = self.etc(torch.cat((modulated, guided, attended), dim=1))

        texture_up_1 = F.interpolate(enhanced, size=texture_1.shape[2:], mode="bilinear", align_corners=False)
        texture_up_1 = self.texture_up_1(torch.cat((texture_up_1, texture_1), dim=1))
        texture_up_2 = F.interpolate(texture_up_1, size=texture_0.shape[2:], mode="bilinear", align_corners=False)
        texture_features = self.texture_up_2(torch.cat((texture_up_2, texture_0), dim=1))
        raw = self.texture_head(texture_features)
        if base_completion is None:
            predicted = torch.sigmoid(raw[:, :3])
        else:
            predicted = (base_completion + 0.25 * torch.tanh(raw[:, :3])).clamp(0.0, 1.0)
        log_variance = raw[:, 3:4].clamp(-6.0, 3.0)
        confidence = torch.sigmoid(-log_variance)
        completed = visibility * partial_uv + (1.0 - visibility) * predicted

        structure_up_1 = F.interpolate(guided, size=structure_1.shape[2:], mode="bilinear", align_corners=False)
        structure_up_1 = self.structure_up_1(torch.cat((structure_up_1, structure_1), dim=1))
        structure_up_2 = F.interpolate(structure_up_1, size=structure_0.shape[2:], mode="bilinear", align_corners=False)
        structure_features = self.structure_up_2(torch.cat((structure_up_2, structure_0), dim=1))
        structure_prediction = torch.sigmoid(self.structure_head(structure_features))
        return StructureGuidedUVOutputs(
            completed_uv=completed,
            predicted_uv=predicted,
            rgb_residual=predicted - partial_uv,
            log_variance=log_variance,
            confidence=confidence,
            structure_prediction=structure_prediction,
            structure_gate=gate,
            cross_attention=attention,
            expression_coefficients=coefficients,
            enhanced_features=enhanced,
        )


class MultiBranchUVDiscriminator(nn.Module if nn is not None else object):
    """Training-only global, structure-boundary, and masked-patch critics."""

    def __init__(self, structure_channels: int = 1, base_channels: int = 16) -> None:
        if torch is None or nn is None:
            raise RuntimeError("structure_guided_uv_requires_torch")
        super().__init__()
        self.global_branch = self._branch(5, base_channels)
        self.structure_branch = self._branch(3 + structure_channels, base_channels)
        self.patch_branch = self._branch(4, base_channels)

    @staticmethod
    def _branch(in_channels: int, base: int):
        return nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, base, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(base, base * 2, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 2, 1),
        )

    def forward(  # type: ignore[override]
        self, uv: Any, structure_boundary: Any, completion_mask: Any
    ) -> MultiBranchDiscriminatorOutputs:
        if uv.ndim != 4 or uv.shape[1] != 3:
            raise ValueError("uv_discriminator_texture_shape_invalid")
        if completion_mask.shape != uv[:, :1].shape:
            raise ValueError("uv_discriminator_mask_shape_invalid")
        height, width = uv.shape[2:]
        y = torch.linspace(-1.0, 1.0, height, device=uv.device, dtype=uv.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=uv.device, dtype=uv.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        position = torch.stack((xx, yy), dim=0)[None].expand(uv.shape[0], -1, -1, -1)
        return MultiBranchDiscriminatorOutputs(
            global_texture=self.global_branch(torch.cat((uv, position), dim=1)),
            structural_boundary=self.structure_branch(torch.cat((uv, structure_boundary), dim=1)),
            masked_patch=self.patch_branch(torch.cat((uv, completion_mask), dim=1)),
        )


def count_structure_guided_uv_parameters(model: Any, *, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (value for value in parameters if value.requires_grad)
    return int(sum(value.numel() for value in parameters))


__all__ = [
    "GatedConvolution",
    "LocalCrossAttention",
    "MultiBranchDiscriminatorOutputs",
    "MultiBranchUVDiscriminator",
    "StructureGuidanceGate",
    "StructureGuidedUVCompletion",
    "StructureGuidedUVConfig",
    "StructureGuidedUVOutputs",
    "count_structure_guided_uv_parameters",
]
