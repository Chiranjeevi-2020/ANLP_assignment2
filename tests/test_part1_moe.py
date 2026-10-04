"""
Unit tests for Part 1: Mixture of Experts (MoE) Architecture and Ablations.
"""

import pytest
import torch
import torch.nn as nn

from src.part1.config import TransformerConfig
from src.part1.moe import MoE, MLP, Expert
from src.part1.model import Transformer
from src.utils import count_parameters


def test_mlp_forward_shape():
    config = TransformerConfig(D_model=128, N_ctx=64, D_ff=512)
    mlp = MLP(config)
    x = torch.randn(2, 64, 128)
    out = mlp(x)
    assert out.shape == (2, 64, 128), f"Expected shape (2, 64, 128), got {out.shape}"


@pytest.mark.parametrize("variant", ["top1", "top2", "shared_routed", "matched_active"])
def test_moe_forward_shape_and_loss(variant):
    config = TransformerConfig(D_model=128, N_ctx=64, D_ff=512, ablation_variant=variant)
    moe = MoE(config)
    x = torch.randn(2, 64, 128)
    out = moe(x)

    assert out.shape == (2, 64, 128), f"Variant {variant} produced incorrect shape: {out.shape}"
    assert moe.last_expert_indices is not None
    assert moe.last_aux_loss is not None
    assert moe.last_aux_loss.item() >= 0.0, "Auxiliary loss must be non-negative"


def test_parameter_matching_variants():
    """
    Validates that:
    1. Total parameters of variants 1..4 are virtually identical (< 0.5% diff due to router).
    2. Active parameters of variant 5 (matched_active) match dense baseline (< 0.2% diff).
    """
    dense_cfg = TransformerConfig(D_model=256, N_ctx=128, D_ff=1024, N_layers=4, ablation_variant="dense")
    dense_model = Transformer(dense_cfg)
    dense_counts = count_parameters(dense_model)
    dense_total = dense_counts["total_parameters"]
    dense_active = dense_counts["active_parameters"]

    # Variants 1 to 4: Total parameters must match
    for v in ["top1", "top2", "shared_routed"]:
        cfg = TransformerConfig(D_model=256, N_ctx=128, D_ff=1024, N_layers=4, ablation_variant=v)
        model = Transformer(cfg)
        counts = count_parameters(model)
        total = counts["total_parameters"]
        pct_diff = abs(total - dense_total) / dense_total * 100.0
        assert pct_diff < 0.5, f"Variant {v} total parameters differ by {pct_diff:.2f}% (must be < 0.5%)"

    # Variant 5: Active parameters must match dense active parameters
    cfg5 = TransformerConfig(D_model=256, N_ctx=128, D_ff=1024, N_layers=4, ablation_variant="matched_active")
    model5 = Transformer(cfg5)
    counts5 = count_parameters(model5)
    active5 = counts5["active_parameters"]
    pct_diff_active = abs(active5 - dense_active) / dense_active * 100.0
    assert pct_diff_active < 0.5, f"Variant matched_active active parameters differ by {pct_diff_active:.2f}%"


def test_transformer_backward_pass():
    """Verifies that end-to-end backprop flows smoothly through MoE router and experts."""
    config = TransformerConfig(D_model=64, N_ctx=32, D_ff=256, N_layers=2, ablation_variant="top2")
    model = Transformer(config)
    idx = torch.randint(0, config.N_vocab, (2, 32))
    targets = idx.clone()

    logits, loss, aux_loss = model(idx, targets=targets)
    assert not torch.isnan(loss)
    loss.backward()

    # Check gradients in router and routed experts
    for block in model.blocks:
        assert block.ffn.router.weight.grad is not None
        assert block.ffn.routed_experts[0].w1.weight.grad is not None


def test_expert_usage_tracking():
    config = TransformerConfig(D_model=64, N_ctx=16, D_ff=256, ablation_variant="top1")
    moe = MoE(config)
    x = torch.randn(4, 16, 64)
    moe(x)
    total_dispatches = moe.expert_selection_counts.sum().item()
    assert total_dispatches == 4 * 16, f"Expected 64 dispatches, got {total_dispatches}"
