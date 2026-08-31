#!/usr/bin/env python3
"""Analytic unit tests for the two CanonReg regularizers."""

from __future__ import annotations

import json

from canonreg_loss import canonical_edge_tv, masked_residual_l2


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"canonreg_loss_test:{label}")


def run_tests() -> dict[str, object]:
    import torch

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(92117)

    factual = torch.rand(2, 3, 5, 6, generator=generator, requires_grad=True)
    base = torch.rand(2, 3, 5, 6, generator=generator)
    visibility = torch.zeros(2, 1, 5, 6)
    canonical = torch.ones(2, 1, 5, 6)
    l2, _support = masked_residual_l2(factual, base, visibility, canonical)
    residual = (factual - base) * (1.0 - visibility)
    require(torch.allclose(l2, residual.square().mean()), "full_grid_l2_equivalence")
    tv, _edges = canonical_edge_tv(factual, base, visibility, canonical)
    expected_tv = (residual[:, :, 1:, :] - residual[:, :, :-1, :]).abs().mean()
    expected_tv = expected_tv + (
        residual[:, :, :, 1:] - residual[:, :, :, :-1]
    ).abs().mean()
    require(torch.allclose(tv, expected_tv), "full_grid_tv_equivalence")

    soft_factual = torch.ones(1, 3, 2, 2)
    soft_base = torch.zeros_like(soft_factual)
    soft_visibility = torch.full((1, 1, 2, 2), 0.5)
    soft_domain = torch.ones(1, 1, 2, 2)
    soft_l2, _ = masked_residual_l2(
        soft_factual, soft_base, soft_visibility, soft_domain
    )
    require(torch.allclose(soft_l2, torch.tensor(0.5)), "soft_mask_residual_definition")

    probe = torch.ones(1, 3, 2, 3, requires_grad=True)
    zero = torch.zeros_like(probe)
    invisible = torch.zeros(1, 1, 2, 3)
    domain = torch.tensor([[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]])
    domain_l2, support = masked_residual_l2(probe, zero, invisible, domain)
    require(torch.allclose(domain_l2, torch.tensor(1.0)), "masked_l2_value")
    require(torch.equal(support, torch.tensor([4.0])), "masked_l2_support")
    domain_l2.backward()
    expanded_domain = domain.bool().expand_as(probe)
    require(torch.equal(probe.grad[~expanded_domain], torch.zeros_like(probe.grad[~expanded_domain])), "outside_domain_zero_gradient")

    visible_left = torch.zeros(1, 1, 2, 3)
    visible_left[:, :, :, 0] = 1.0
    all_domain = torch.ones(1, 1, 2, 3)
    boundary_tv, _ = canonical_edge_tv(probe.detach(), zero, visible_left, all_domain)
    require(torch.allclose(boundary_tv, torch.tensor(0.5)), "visible_hidden_boundary_retained")
    clipped_tv, clipped_edges = canonical_edge_tv(probe.detach() * torch.tensor([[[[1.0, 1.0, 100.0], [1.0, 1.0, 100.0]]]]), zero, visible_left, domain)
    require(torch.allclose(clipped_tv, torch.tensor(1.0)), "outside_domain_edge_excluded")
    require(torch.equal(clipped_edges["canonical_edges_x"], torch.tensor([2.0])), "horizontal_edge_count")
    require(torch.equal(clipped_edges["canonical_edges_y"], torch.tensor([2.0])), "vertical_edge_count")

    return {
        "status": "PASS_CANONREG_LOSS_UNIT_TESTS",
        "full_grid_equivalence": True,
        "soft_mask_residual_definition": True,
        "outside_domain_zero_gradient": True,
        "visible_hidden_boundary_retained": True,
        "outside_domain_edges_excluded": True,
        "scientific_result_generated": False,
    }


def main() -> int:
    print(json.dumps(run_tests(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
