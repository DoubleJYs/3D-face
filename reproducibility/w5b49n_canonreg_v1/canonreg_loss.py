"""Canon-domain residual and TV regularizers; all other Full losses stay fixed."""

from __future__ import annotations

from typing import Any, Mapping


def _per_sample_denominator(mask: Any, channels: int, label: str) -> Any:
    denominator = mask.sum(dim=(1, 2, 3)) * float(channels)
    if bool((denominator <= 0).any()):
        raise RuntimeError(f"canonreg_empty_support:{label}")
    return denominator


def masked_residual_l2(
    factual: Any,
    base_completion: Any,
    visibility: Any,
    canonical_mask: Any,
) -> tuple[Any, Any]:
    """L2 over Mcanon*(1-Vs), normalized by valid texels times RGB channels."""

    support = canonical_mask * (1.0 - visibility)
    denominator = _per_sample_denominator(support, factual.shape[1], "residual_l2")
    residual = (factual - base_completion) * (1.0 - visibility)
    squared = residual.square() * canonical_mask
    per_sample = squared.sum(dim=(1, 2, 3)) / denominator
    return per_sample.mean(), support.sum(dim=(1, 2, 3))


def canonical_edge_tv(
    factual: Any,
    base_completion: Any,
    visibility: Any,
    canonical_mask: Any,
) -> tuple[Any, dict[str, Any]]:
    """Anisotropic TV on edges whose two endpoints lie inside Mcanon.

    The residual field is zero on source-visible texels. Consequently, an edge
    crossing a visible-hidden boundary inside Mcanon remains part of the loss.
    """

    residual = (factual - base_completion) * (1.0 - visibility)
    edge_y = canonical_mask[:, :, 1:, :] * canonical_mask[:, :, :-1, :]
    edge_x = canonical_mask[:, :, :, 1:] * canonical_mask[:, :, :, :-1]
    denominator_y = _per_sample_denominator(edge_y, factual.shape[1], "tv_y")
    denominator_x = _per_sample_denominator(edge_x, factual.shape[1], "tv_x")
    difference_y = (residual[:, :, 1:, :] - residual[:, :, :-1, :]).abs()
    difference_x = (residual[:, :, :, 1:] - residual[:, :, :, :-1]).abs()
    per_sample_y = (difference_y * edge_y).sum(dim=(1, 2, 3)) / denominator_y
    per_sample_x = (difference_x * edge_x).sum(dim=(1, 2, 3)) / denominator_x
    return per_sample_y.mean() + per_sample_x.mean(), {
        "canonical_edges_y": edge_y.sum(dim=(1, 2, 3)),
        "canonical_edges_x": edge_x.sum(dim=(1, 2, 3)),
    }


def compute_training_loss(
    historical_core: Any,
    model: Any,
    batch: Mapping[str, Any],
    *,
    anchor_count: int,
) -> tuple[Any, dict[str, Any], Any, dict[str, Any]]:
    """Full loss with exactly two review-triggered regularizer substitutions."""

    import torch
    import torch.nn.functional as F

    output = historical_core.forward_matched(
        model, historical_core.METHOD_FULL, batch
    )
    anchor = {name: value[:anchor_count] for name, value in batch.items()}
    paired = {
        name: value[anchor_count : 2 * anchor_count]
        for name, value in batch.items()
    }
    mask = historical_core.hidden_mask(anchor, paired)
    if bool((mask.sum(dim=(1, 2, 3)) < 1.0).any()):
        raise RuntimeError("canonreg_hidden_support_changed")
    factual = output.completed_uv[:anchor_count]
    target = paired["partial_uv"]
    hidden_l1 = historical_core.masked_l1(factual, target, mask).mean()
    rgb_error = ((factual - target).abs() * mask).sum(dim=1, keepdim=True) / 3.0
    nll = (
        (
            rgb_error * torch.exp(-output.log_variance[:anchor_count])
            + output.log_variance[:anchor_count]
        )
        * mask
    ).sum() / mask.sum().clamp_min(1.0)
    canonical = anchor["canonical_mask"]
    view_consistency = (
        (
            output.completed_uv[:anchor_count]
            - output.completed_uv[anchor_count : 2 * anchor_count]
        ).abs()
        * canonical
    ).sum() / (canonical.sum().clamp_min(1.0) * 3.0)
    positive = F.cosine_similarity(
        output.identity_embedding[:anchor_count],
        output.identity_embedding[anchor_count : 2 * anchor_count],
    )
    negative = F.cosine_similarity(
        output.identity_embedding[:anchor_count],
        output.identity_embedding[2 * anchor_count : 3 * anchor_count],
    )
    triplet = F.relu(
        historical_core.IDENTITY_TRIPLET_MARGIN - positive + negative
    ).mean()
    structure_mean = output.structure_gate[:anchor_count, 1].mean()
    gate_noncollapse = F.relu(0.10 - structure_mean) + F.relu(
        structure_mean - 0.90
    )
    residual_l2, residual_support = masked_residual_l2(
        factual,
        anchor["base_completion"],
        anchor["visibility"],
        anchor["canonical_mask"],
    )
    tv, edge_counts = canonical_edge_tv(
        factual,
        anchor["base_completion"],
        anchor["visibility"],
        anchor["canonical_mask"],
    )
    terms = {
        "paired_hidden_UV_L1": hidden_l1,
        "paired_hidden_uncertainty_NLL": nll,
        "same_identity_view_consistency": view_consistency,
        "identity_triplet": triplet,
        "structure_gate_noncollapse": gate_noncollapse,
        "bounded_residual_L2": residual_l2,
        "total_variation": tv,
    }
    loss = sum(
        historical_core.LOSS_WEIGHTS[name] * value for name, value in terms.items()
    )
    diagnostics = {
        "residual_support_texels": residual_support,
        **edge_counts,
    }
    return loss, terms, output, diagnostics


__all__ = [
    "canonical_edge_tv",
    "compute_training_loss",
    "masked_residual_l2",
]
