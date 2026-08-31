"""Dynamic full-parent-B identity evidence for one-loop A-to-B-to-A fusion.

The already trained UV completion expert is an immutable appearance anchor.
This module restores the missing structure branch, softmax SGG,
texture-query/structure-key-value local attention, ETC and uncertainty.  The
Geometry-A feature may only generate low-rank modulation parameters for B; it
has no direct path to the FLAME shape head.  Consequently, removing B is an
exact Geometry-A fallback and swapping B is a genuine causal intervention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - construction requires torch.
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover - keep repository audits import-safe.
    torch = None
    F = None
    nn = None

from .structure_guided_uv_completion import LocalCrossAttention


DYNAMIC_PARENT_B_ROUTES = {
    "factual",
    "cross_identity_b",
    "spatial_shuffle_b",
    "b_absent",
}


@dataclass(frozen=True)
class DynamicParentBIdentityConfig:
    a_feature_dim: int = 2048
    texture_feature_channels: int = 160
    geometry_channels: int = 6
    hidden_channels: int = 64
    structure_stem_channels: int = 32
    dynamic_rank: int = 8
    shape_rank: int = 16
    shape_dim: int = 100
    active_shape_dim: int = 100
    pooling_grid: int = 4
    attention_window: int = 3
    maximum_dynamic_modulation: float = 0.5
    maximum_texture_residual: float = 0.10
    maximum_shape_delta: float = 0.15
    maximum_parameters: int = 250_000


@dataclass
class DynamicParentBIdentityOutput:
    shape: Any
    shape_delta: Any
    completed_uv: Any | None
    confidence: Any | None
    uncertainty: Any | None
    structure_gate: Any | None
    cross_attention: Any | None
    enhanced_features: Any | None
    dynamic_coefficients: Any | None
    trust: Any
    route: str
    exact_fallback: bool


def _require_torch() -> None:
    if torch is None or F is None or nn is None:
        raise RuntimeError("dynamic_parent_b_identity_requires_torch")


def _group_norm(channels: int):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SoftmaxStructureGuidanceGate(nn.Module if nn is not None else object):
    """Two-source SGG with an explicit texture/structure softmax contract."""

    def __init__(self, channels: int) -> None:
        _require_torch()
        super().__init__()
        self.logits = nn.Conv2d(2 * channels, 2, 1)

    def forward(self, texture: Any, structure: Any) -> tuple[Any, Any]:  # type: ignore[override]
        weights = torch.softmax(self.logits(torch.cat((texture, structure), dim=1)), dim=1)
        guided = weights[:, :1] * texture + weights[:, 1:2] * structure
        return guided, weights


class DynamicParentBIdentityFeedback(nn.Module if nn is not None else object):
    """Versioned parent-B evidence bridge with no A-to-shape shortcut."""

    def __init__(self, config: DynamicParentBIdentityConfig | None = None) -> None:
        _require_torch()
        super().__init__()
        self.config = config or DynamicParentBIdentityConfig()
        cfg = self.config
        if not 1 <= cfg.active_shape_dim <= cfg.shape_dim:
            raise ValueError("dynamic_parent_b_active_shape_invalid")
        if not 2 <= cfg.dynamic_rank <= 16 or not 2 <= cfg.shape_rank <= 32:
            raise ValueError("dynamic_parent_b_rank_invalid")
        if cfg.hidden_channels % 8 or cfg.pooling_grid < 2:
            raise ValueError("dynamic_parent_b_hidden_contract_invalid")

        hidden = cfg.hidden_channels
        stem = cfg.structure_stem_channels
        self.texture_projection = nn.Sequential(
            nn.Conv2d(cfg.texture_feature_channels, hidden, 1, bias=False),
            _group_norm(hidden),
            nn.SiLU(inplace=True),
        )
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(cfg.geometry_channels + 2, stem, 3, stride=2, padding=1, bias=False),
            _group_norm(stem),
            nn.SiLU(inplace=True),
            nn.Conv2d(stem, hidden, 3, stride=2, padding=1, bias=False),
            _group_norm(hidden),
            nn.SiLU(inplace=True),
        )
        self.sgg = SoftmaxStructureGuidanceGate(hidden)
        self.cross_attention = LocalCrossAttention(hidden, cfg.attention_window)

        # Geometry A generates only coefficients over a small B-parameter basis.
        # There is deliberately no A feature in either shape head below.
        self.a_dynamic_coefficients = nn.Linear(cfg.a_feature_dim, cfg.dynamic_rank)
        self.dynamic_basis = nn.Parameter(torch.empty(cfg.dynamic_rank, 2 * hidden))
        nn.init.normal_(self.dynamic_basis, mean=0.0, std=0.01)

        self.etc = nn.Sequential(
            nn.Conv2d(3 * hidden, hidden, 1, bias=False),
            _group_norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            _group_norm(hidden),
            nn.SiLU(inplace=True),
        )
        self.uncertainty_head = nn.Conv2d(hidden, 1, 1)
        self.texture_residual_head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            _group_norm(hidden // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden // 2, 3, 1),
        )
        nn.init.zeros_(self.texture_residual_head[-1].weight)
        nn.init.zeros_(self.texture_residual_head[-1].bias)
        pooled = hidden * cfg.pooling_grid * cfg.pooling_grid
        self.shape_coefficients = nn.Sequential(
            nn.Linear(pooled, hidden),
            nn.SiLU(),
            nn.Linear(hidden, cfg.shape_rank),
        )
        self.shape_basis = nn.Parameter(torch.empty(cfg.shape_rank, cfg.active_shape_dim))
        self.trust_head = nn.Sequential(
            nn.Linear(pooled, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.shape_basis, mean=0.0, std=0.01)
        if self.parameter_count > cfg.maximum_parameters:
            raise ValueError(
                "dynamic_parent_b_parameter_budget_exceeded:"
                f"{self.parameter_count}>{cfg.maximum_parameters}"
            )

    @property
    def parameter_count(self) -> int:
        return int(sum(value.numel() for value in self.parameters() if value.requires_grad))

    def _fallback(self, base_shape: Any, route: str) -> DynamicParentBIdentityOutput:
        return DynamicParentBIdentityOutput(
            shape=base_shape,
            shape_delta=torch.zeros_like(base_shape),
            completed_uv=None,
            confidence=None,
            uncertainty=None,
            structure_gate=None,
            cross_attention=None,
            enhanced_features=None,
            dynamic_coefficients=None,
            trust=base_shape.new_zeros((base_shape.shape[0], 1)),
            route=route,
            exact_fallback=True,
        )

    def _validate(
        self,
        base_shape: Any,
        a_feature: Any,
        texture_feature: Any,
        completed_uv: Any,
        b_confidence: Any,
        geometry_map: Any,
        visibility: Any,
        canonical_mask: Any,
    ) -> None:
        cfg = self.config
        batch = int(base_shape.shape[0])
        expected = {
            "base_shape": (batch, cfg.shape_dim),
            "a_feature": (batch, cfg.a_feature_dim),
            "texture_feature": (batch, cfg.texture_feature_channels, texture_feature.shape[2], texture_feature.shape[3]),
            "completed_uv": (batch, 3, geometry_map.shape[2], geometry_map.shape[3]),
            "b_confidence": (batch, 1, geometry_map.shape[2], geometry_map.shape[3]),
            "geometry_map": (batch, cfg.geometry_channels, geometry_map.shape[2], geometry_map.shape[3]),
            "visibility": (batch, 1, geometry_map.shape[2], geometry_map.shape[3]),
            "canonical_mask": (batch, 1, geometry_map.shape[2], geometry_map.shape[3]),
        }
        values = locals()
        for name, shape in expected.items():
            value = values[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"dynamic_parent_b_{name}_shape_invalid")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"dynamic_parent_b_{name}_nonfinite")
        if texture_feature.shape[2:] != (
            geometry_map.shape[2] // 4,
            geometry_map.shape[3] // 4,
        ):
            raise ValueError("dynamic_parent_b_texture_feature_scale_invalid")
        for name in ("b_confidence", "visibility", "canonical_mask"):
            value = values[name]
            if bool((value < 0).any()) or bool((value > 1).any()):
                raise ValueError(f"dynamic_parent_b_{name}_range_invalid")

    @staticmethod
    def _donate(value: Any, indices: Any | None) -> Any:
        if indices is None:
            raise ValueError("dynamic_parent_b_cross_identity_donor_missing")
        if tuple(indices.shape) != (value.shape[0],):
            raise ValueError("dynamic_parent_b_cross_identity_donor_shape_invalid")
        return value.index_select(0, indices.long())

    def forward(  # type: ignore[override]
        self,
        base_shape: Any,
        a_feature: Any,
        texture_feature: Any | None,
        completed_uv: Any | None,
        b_confidence: Any | None,
        geometry_map: Any | None,
        visibility: Any | None,
        canonical_mask: Any | None,
        *,
        route: str = "factual",
        identity_donor_indices: Any | None = None,
    ) -> DynamicParentBIdentityOutput:
        if route not in DYNAMIC_PARENT_B_ROUTES:
            raise ValueError(f"dynamic_parent_b_unknown_route:{route}")
        if route == "b_absent" or any(
            value is None
            for value in (
                texture_feature,
                completed_uv,
                b_confidence,
                geometry_map,
                visibility,
                canonical_mask,
            )
        ):
            return self._fallback(base_shape, route)
        self._validate(
            base_shape,
            a_feature,
            texture_feature,
            completed_uv,
            b_confidence,
            geometry_map,
            visibility,
            canonical_mask,
        )
        if route == "cross_identity_b":
            texture_feature = self._donate(texture_feature, identity_donor_indices)
            completed_uv = self._donate(completed_uv, identity_donor_indices)
            b_confidence = self._donate(b_confidence, identity_donor_indices)
            geometry_map = self._donate(geometry_map, identity_donor_indices)
            visibility = self._donate(visibility, identity_donor_indices)
            canonical_mask = self._donate(canonical_mask, identity_donor_indices)
        elif route == "spatial_shuffle_b":
            texture_feature = texture_feature.flip(dims=(2, 3))
            completed_uv = completed_uv.flip(dims=(2, 3))
            b_confidence = b_confidence.flip(dims=(2, 3))
            geometry_map = geometry_map.flip(dims=(2, 3))
            visibility = visibility.flip(dims=(2, 3))
            canonical_mask = canonical_mask.flip(dims=(2, 3))

        texture = self.texture_projection(texture_feature)
        structure = self.structure_encoder(
            torch.cat((geometry_map, visibility, canonical_mask), dim=1)
        )
        guided, gate = self.sgg(texture, structure)
        attended, attention = self.cross_attention(texture, structure)
        dynamic_coefficients = torch.softmax(self.a_dynamic_coefficients(a_feature), dim=1)
        gamma, beta = (dynamic_coefficients @ self.dynamic_basis).chunk(2, dim=1)
        dynamic = guided * (
            1.0
            + self.config.maximum_dynamic_modulation
            * torch.tanh(gamma)[:, :, None, None]
        )
        dynamic = dynamic + self.config.maximum_dynamic_modulation * torch.tanh(beta)[
            :, :, None, None
        ]
        enhanced = self.etc(torch.cat((texture, dynamic, attended), dim=1))
        uncertainty = self.uncertainty_head(enhanced).clamp(-6.0, 3.0)
        texture_residual = F.interpolate(
            self.texture_residual_head(enhanced),
            completed_uv.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        predicted_uv = (
            completed_uv
            + self.config.maximum_texture_residual * torch.tanh(texture_residual)
        ).clamp(0.0, 1.0)
        routed_uv = visibility * completed_uv + (1.0 - visibility) * predicted_uv
        expert_confidence = F.adaptive_avg_pool2d(b_confidence, enhanced.shape[2:])
        confidence = torch.sigmoid(-uncertainty) * expert_confidence
        mask = F.interpolate(canonical_mask, enhanced.shape[2:], mode="nearest")
        weights = confidence * mask
        grid = (self.config.pooling_grid, self.config.pooling_grid)
        numerator = F.adaptive_avg_pool2d(enhanced * weights, grid)
        denominator = F.adaptive_avg_pool2d(weights, grid).clamp_min(1e-6)
        pooled = (numerator / denominator).flatten(1)
        evidence_mass = weights.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        trust = self.trust_head(pooled) * evidence_mass.clamp(0.0, 1.0)
        coefficients = torch.tanh(self.shape_coefficients(pooled))
        active_delta = (
            trust
            * self.config.maximum_shape_delta
            * torch.tanh(coefficients @ self.shape_basis)
        )
        delta = torch.zeros_like(base_shape)
        delta[:, : self.config.active_shape_dim] = active_delta
        return DynamicParentBIdentityOutput(
            shape=base_shape + delta,
            shape_delta=delta,
            completed_uv=routed_uv,
            confidence=confidence,
            uncertainty=uncertainty,
            structure_gate=gate,
            cross_attention=attention,
            enhanced_features=enhanced,
            dynamic_coefficients=dynamic_coefficients,
            trust=trust,
            route=route,
            exact_fallback=False,
        )


__all__ = [
    "DYNAMIC_PARENT_B_ROUTES",
    "DynamicParentBIdentityConfig",
    "DynamicParentBIdentityFeedback",
    "DynamicParentBIdentityOutput",
    "SoftmaxStructureGuidanceGate",
]
