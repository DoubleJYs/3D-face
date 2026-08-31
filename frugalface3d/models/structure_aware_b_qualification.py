"""Warm-started structure-aware B used by the W5-B-45B qualification gate.

The frozen B-lite expert supplies the appearance prior and its bottleneck.  This
module is the trainable structure-aware residual path.  It never updates
Geometry A and preserves every visible UV texel exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - construction requires torch.
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    F = None
    nn = None

from .dynamic_parent_b_identity_feedback import SoftmaxStructureGuidanceGate
from .structure_guided_uv_completion import LocalCrossAttention


def _group_norm(channels: int) -> Any:
    if nn is None:
        raise RuntimeError("structure_aware_b_requires_torch")
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


@dataclass(frozen=True)
class StructureAwareBQualificationConfig:
    texture_feature_channels: int = 160
    structure_channels: int = 6
    expression_token_dim: int = 128
    hidden_channels: int = 64
    dynamic_rank: int = 4
    attention_window: int = 3
    maximum_uv_residual: float = 0.15
    maximum_parameters: int = 250_000


@dataclass
class StructureAwareBQualificationOutputs:
    completed_uv: Any
    predicted_uv: Any
    log_variance: Any
    confidence: Any
    structure_gate: Any
    cross_attention: Any
    expression_coefficients: Any
    identity_embedding: Any
    enhanced_features: Any


class StructureAwareBQualification(nn.Module if nn is not None else object):
    """Texture-query/structure-key-value residual completion over frozen B-lite."""

    def __init__(self, config: StructureAwareBQualificationConfig | None = None) -> None:
        if torch is None or nn is None or F is None:
            raise RuntimeError("structure_aware_b_requires_torch")
        super().__init__()
        self.config = config or StructureAwareBQualificationConfig()
        cfg = self.config
        if cfg.hidden_channels < 16 or cfg.dynamic_rank < 2:
            raise ValueError("structure_aware_b_capacity_contract_invalid")
        hidden = cfg.hidden_channels
        self.texture_project = nn.Sequential(
            nn.Conv2d(cfg.texture_feature_channels, hidden, 1, bias=False),
            _group_norm(hidden),
            nn.SiLU(inplace=True),
        )
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(cfg.structure_channels + 2, hidden // 2, 3, stride=2, padding=1, bias=False),
            _group_norm(hidden // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden // 2, hidden, 3, stride=2, padding=1, bias=False),
            _group_norm(hidden),
            nn.SiLU(inplace=True),
        )
        self.sgg = SoftmaxStructureGuidanceGate(hidden)
        self.cross_attention = LocalCrossAttention(hidden, cfg.attention_window)
        self.expression_coefficients = nn.Linear(cfg.expression_token_dim, cfg.dynamic_rank)
        self.expression_basis = nn.Parameter(torch.zeros(cfg.dynamic_rank, hidden * 2))
        nn.init.normal_(self.expression_basis, mean=0.0, std=0.01)
        self.etc = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1, bias=False),
            _group_norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            _group_norm(hidden),
            nn.SiLU(inplace=True),
        )
        self.up_1 = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            _group_norm(hidden // 2),
            nn.SiLU(inplace=True),
        )
        self.up_2 = nn.Sequential(
            nn.Conv2d(hidden // 2, hidden // 4, 3, padding=1, bias=False),
            _group_norm(hidden // 4),
            nn.SiLU(inplace=True),
        )
        self.output_head = nn.Conv2d(hidden // 4, 4, 1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        if self.parameter_count > cfg.maximum_parameters:
            raise ValueError(
                f"structure_aware_b_parameter_budget_exceeded:{self.parameter_count}>"
                f"{cfg.maximum_parameters}"
            )

    @property
    def parameter_count(self) -> int:
        return int(sum(value.numel() for value in self.parameters() if value.requires_grad))

    def _validate(
        self,
        partial_uv: Any,
        visibility: Any,
        geometry_map: Any,
        canonical_mask: Any,
        base_completion: Any,
        texture_feature: Any,
        expression_token: Any,
    ) -> None:
        batch, _, height, width = partial_uv.shape
        expected = {
            "visibility": ((batch, 1, height, width), visibility),
            "geometry_map": ((batch, self.config.structure_channels, height, width), geometry_map),
            "canonical_mask": ((batch, 1, height, width), canonical_mask),
            "base_completion": ((batch, 3, height, width), base_completion),
            "texture_feature": (
                (batch, self.config.texture_feature_channels, height // 4, width // 4),
                texture_feature,
            ),
            "expression_token": ((batch, self.config.expression_token_dim), expression_token),
        }
        if partial_uv.ndim != 4 or partial_uv.shape[1] != 3 or height % 4 or width % 4:
            raise ValueError("structure_aware_b_partial_uv_contract_invalid")
        for name, (shape, value) in expected.items():
            if tuple(value.shape) != shape:
                raise ValueError(f"structure_aware_b_{name}_shape_invalid:{tuple(value.shape)}!={shape}")
        values = (partial_uv, visibility, geometry_map, canonical_mask, base_completion,
                  texture_feature, expression_token)
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("structure_aware_b_nonfinite_input")

    def forward(  # type: ignore[override]
        self,
        partial_uv: Any,
        visibility: Any,
        geometry_map: Any,
        canonical_mask: Any,
        base_completion: Any,
        texture_feature: Any,
        expression_token: Any,
    ) -> StructureAwareBQualificationOutputs:
        self._validate(
            partial_uv,
            visibility,
            geometry_map,
            canonical_mask,
            base_completion,
            texture_feature,
            expression_token,
        )
        texture = self.texture_project(texture_feature)
        structure = self.structure_encoder(
            torch.cat((geometry_map, visibility, canonical_mask), dim=1)
        )
        guided, gate = self.sgg(texture, structure)
        attended, attention = self.cross_attention(texture, guided)
        coefficients = torch.softmax(self.expression_coefficients(expression_token), dim=1)
        gamma, beta = (coefficients @ self.expression_basis).chunk(2, dim=1)
        modulated = texture * (1.0 + 0.25 * torch.tanh(gamma)[:, :, None, None])
        modulated = modulated + 0.25 * torch.tanh(beta)[:, :, None, None]
        enhanced = self.etc(torch.cat((modulated, guided, attended), dim=1))
        decoded = F.interpolate(enhanced, scale_factor=2, mode="bilinear", align_corners=False)
        decoded = self.up_1(decoded)
        decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
        raw = self.output_head(self.up_2(decoded))
        residual = self.config.maximum_uv_residual * torch.tanh(raw[:, :3])
        predicted = (base_completion + (1.0 - visibility) * residual).clamp(0.0, 1.0)
        completed = visibility * partial_uv + (1.0 - visibility) * predicted
        log_variance = raw[:, 3:4].clamp(-6.0, 3.0)
        confidence = torch.sigmoid(-log_variance)
        identity = F.normalize(F.adaptive_avg_pool2d(enhanced, 1).flatten(1), dim=1)
        return StructureAwareBQualificationOutputs(
            completed_uv=completed,
            predicted_uv=predicted,
            log_variance=log_variance,
            confidence=confidence,
            structure_gate=gate,
            cross_attention=attention,
            expression_coefficients=coefficients,
            identity_embedding=identity,
            enhanced_features=enhanced,
        )


__all__ = [
    "StructureAwareBQualification",
    "StructureAwareBQualificationConfig",
    "StructureAwareBQualificationOutputs",
]
