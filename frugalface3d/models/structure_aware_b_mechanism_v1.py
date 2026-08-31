"""Matched FrugalFace3D mechanism variants for the W49N V1 closure.

This module is append-only: the historical
``StructureAwareBQualification`` implementation remains unchanged.  ``Full``
is byte-for-byte state-dict compatible with that implementation, while each
other variant changes exactly one mechanism named by the frozen V1 protocol.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

try:  # pragma: no cover - construction requires torch.
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from .structure_aware_b_qualification import (
    StructureAwareBQualification,
    StructureAwareBQualificationConfig,
    _group_norm,
)


class MechanismVariant(str, Enum):
    FULL = "Full"
    XYZ0 = "XYZ0"
    NORMAL0 = "Normal0"
    EXPRESSION0 = "Expression0"
    XYZ_NORMAL0 = "XYZNormal0"
    GATE_EQUAL = "GateEqual"
    CA_CONV = "CAConv"
    ETC_PLAIN = "ETCPlain"


V1_VARIANTS: tuple[MechanismVariant, ...] = tuple(MechanismVariant)


class _EqualStructureGuidanceGate(nn.Module if nn is not None else object):
    """Parameter-free 50/50 texture--structure fusion."""

    def __init__(self) -> None:
        if nn is None:
            raise RuntimeError("mechanism_variants_require_torch")
        super().__init__()

    def forward(self, texture: Any, structure: Any) -> tuple[Any, Any]:  # type: ignore[override]
        batch, _, height, width = texture.shape
        weights = texture.new_full((batch, 2, height, width), 0.5)
        return 0.5 * (texture + structure), weights


class _CapacityMatchedLocalConv(nn.Module if nn is not None else object):
    """Local convolutional fusion replacing query/key/value attention.

    The three ordinary convolutions keep the complete model within five per
    cent of Full.  A uniform diagnostic map is returned only to preserve the
    historical telemetry interface; it is not used in the forward result.
    """

    def __init__(self, channels: int, window: int) -> None:
        if nn is None:
            raise RuntimeError("mechanism_variants_require_torch")
        super().__init__()
        if window < 1 or window % 2 == 0:
            raise ValueError("mechanism_ca_conv_window_must_be_positive_odd")
        self.window = int(window)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            _group_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=8, bias=False),
            _group_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, texture: Any, structure: Any) -> tuple[Any, Any]:  # type: ignore[override]
        batch, _, height, width = texture.shape
        attended = self.fusion(torch.cat((texture, structure), dim=1))
        neighbors = self.window * self.window
        diagnostic = texture.new_full(
            (batch, neighbors, height, width), 1.0 / float(neighbors)
        )
        return attended, diagnostic


def _capacity_matched_plain_etc(hidden: int) -> Any:
    """Ordinary bottleneck convolution block matching ETC capacity."""

    if nn is None:
        raise RuntimeError("mechanism_variants_require_torch")
    mid = 48 if hidden == 64 else max(8, (hidden * 3) // 4)
    low = 16 if hidden == 64 else max(4, hidden // 4)
    return nn.Sequential(
        nn.Conv2d(hidden * 3, mid, 1, bias=False),
        _group_norm(mid),
        nn.SiLU(inplace=True),
        nn.Conv2d(mid, low, 3, padding=1, bias=False),
        _group_norm(low),
        nn.SiLU(inplace=True),
        nn.Conv2d(low, hidden, 1, bias=False),
        _group_norm(hidden),
        nn.SiLU(inplace=True),
    )


class StructureAwareBMechanismV1(StructureAwareBQualification):
    """One frozen-contract mechanism variant with the historical interface."""

    def __init__(
        self,
        config: StructureAwareBQualificationConfig | None = None,
        *,
        variant: MechanismVariant | str = MechanismVariant.FULL,
    ) -> None:
        self.variant = MechanismVariant(variant)
        super().__init__(config)
        hidden = self.config.hidden_channels
        if self.variant is MechanismVariant.GATE_EQUAL:
            self.sgg = _EqualStructureGuidanceGate()
        elif self.variant is MechanismVariant.CA_CONV:
            self.cross_attention = _CapacityMatchedLocalConv(
                hidden, self.config.attention_window
            )
        elif self.variant is MechanismVariant.ETC_PLAIN:
            self.etc = _capacity_matched_plain_etc(hidden)

    def intervened_inputs(
        self, geometry_map: Any, expression_token: Any
    ) -> tuple[Any, Any]:
        """Apply the single registered condition intervention."""

        if torch is None:
            raise RuntimeError("mechanism_variants_require_torch")
        geometry = geometry_map
        expression = expression_token
        if self.variant in (MechanismVariant.XYZ0, MechanismVariant.XYZ_NORMAL0):
            geometry = geometry.clone()
            geometry[:, :3] = 0
        if self.variant in (MechanismVariant.NORMAL0, MechanismVariant.XYZ_NORMAL0):
            if geometry is geometry_map:
                geometry = geometry.clone()
            geometry[:, 3:6] = 0
        if self.variant is MechanismVariant.EXPRESSION0:
            expression = torch.zeros_like(expression_token)
        return geometry, expression

    def forward(  # type: ignore[override]
        self,
        partial_uv: Any,
        visibility: Any,
        geometry_map: Any,
        canonical_mask: Any,
        base_completion: Any,
        texture_feature: Any,
        expression_token: Any,
    ) -> Any:
        geometry, expression = self.intervened_inputs(geometry_map, expression_token)
        return super().forward(
            partial_uv,
            visibility,
            geometry,
            canonical_mask,
            base_completion,
            texture_feature,
            expression,
        )


def parameter_report(
    config: StructureAwareBQualificationConfig | None = None,
) -> dict[str, int]:
    """Return registered trainable parameters for all eight V1 structures."""

    return {
        variant.value: StructureAwareBMechanismV1(config, variant=variant).parameter_count
        for variant in V1_VARIANTS
    }


__all__ = [
    "MechanismVariant",
    "StructureAwareBMechanismV1",
    "V1_VARIANTS",
    "parameter_report",
]
