"""
Unit tests for Part 2: Custom Optimizers from Scratch.
Verifies that all 5 custom optimizers strictly inherit from torch.optim.Optimizer
and converge on quadratic optimization objectives.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.part2.optimizers import (
    CustomAdamW,
    CustomAdan,
    CustomLion,
    CustomMuon,
    CustomSophia,
)


@pytest.mark.parametrize("OptClass,lr,epochs", [
    (CustomAdamW, 0.1, 70),
    (CustomAdan, 0.05, 50),
    (CustomLion, 0.05, 80),
    (CustomMuon, 0.05, 60),
    (CustomSophia, 0.05, 50),
])
def test_optimizer_convergence_on_quadratic(OptClass, lr, epochs):
    """
    Tests that the optimizer minimizes a convex quadratic objective f(w) = 0.5 * ||w - target||^2.
    """
    torch.manual_seed(42)
    dim = 8
    target = torch.randn(dim, dim)

    # 2D parameter tensor (dim x dim matrix) compatible with all optimizers including Muon
    w = nn.Parameter(torch.randn(dim, dim) * 0.5)
    optimizer = OptClass([w], lr=lr, weight_decay=0.0)

    # Check inheritance
    assert issubclass(OptClass, Optimizer), f"{OptClass} must inherit from torch.optim.Optimizer"

    initial_loss = 0.5 * torch.sum((w - target) ** 2).item()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = 0.5 * torch.sum((w - target) ** 2)
        loss.backward()
        if OptClass == CustomSophia:
            optimizer.update_hessian()
        optimizer.step()

    final_loss = 0.5 * torch.sum((w - target) ** 2).item()
    assert final_loss < initial_loss * 0.1, f"{OptClass.__name__} failed to minimize loss: {initial_loss:.4f} -> {final_loss:.4f}"


def test_muon_newton_schulz_orthogonality():
    """Verifies that Muon's Newton-Schulz iteration outputs an approximately orthogonal matrix."""
    from src.part2.optimizers import newton_schulz5

    torch.manual_seed(42)
    G = torch.randn(8, 8)
    O = newton_schulz5(G, steps=6)

    # For orthogonal matrix O, O @ O.T should be close to identity
    gram = O @ O.T
    off_diag = gram - torch.diag(torch.diag(gram))
    off_diag_norm = off_diag.norm().item()

    assert off_diag_norm < 1.0, f"Off-diagonal elements too large in orthogonalized matrix: {off_diag_norm}"


def test_decoupled_weight_decay():
    """Verifies that decoupled weight decay reduces weights even with zero gradient."""
    p = nn.Parameter(torch.ones(4, 4))
    opt = CustomAdamW([p], lr=0.1, weight_decay=0.1)

    p.grad = torch.zeros_like(p)
    opt.step()

    # With decoupled weight decay, p = p * (1 - lr * wd) = 1.0 * (1 - 0.01) = 0.99
    assert torch.allclose(p, torch.full((4, 4), 0.99), atol=1e-5), f"Decoupled weight decay mismatch: {p[0, 0].item()}"


def test_muon_mixed_1d_and_2d_parameters():
    """Verifies that Muon handles both 2D weight matrices and 1D vector parameters simultaneously."""
    torch.manual_seed(42)
    w2d = nn.Parameter(torch.randn(8, 8))
    b1d = nn.Parameter(torch.randn(8))
    opt = CustomMuon([w2d, b1d], lr=0.05, fallback_lr=0.05, weight_decay=0.01)

    target2d = torch.randn(8, 8)
    target1d = torch.randn(8)

    for _ in range(50):
        opt.zero_grad()
        loss = 0.5 * torch.sum((w2d - target2d) ** 2) + 0.5 * torch.sum((b1d - target1d) ** 2)
        loss.backward()
        opt.step()

    final_loss = 0.5 * torch.sum((w2d - target2d) ** 2) + 0.5 * torch.sum((b1d - target1d) ** 2)
    assert not torch.isnan(final_loss)
    assert final_loss.item() < 30.0


def test_sophia_without_update_hessian_converges():
    """Verifies that CustomSophia converges even when update_hessian is never explicitly called."""
    torch.manual_seed(42)
    dim = 8
    target = torch.randn(dim, dim)
    w = nn.Parameter(torch.randn(dim, dim) * 0.5)
    opt = CustomSophia([w], lr=0.05, weight_decay=0.0)

    initial_loss = 0.5 * torch.sum((w - target) ** 2).item()
    for _ in range(50):
        opt.zero_grad()
        loss = 0.5 * torch.sum((w - target) ** 2)
        loss.backward()
        opt.step()  # Note: NOT calling update_hessian()

    final_loss = 0.5 * torch.sum((w - target) ** 2).item()
    assert final_loss < initial_loss * 0.1, f"CustomSophia failed to converge without update_hessian: {initial_loss:.4f} -> {final_loss:.4f}"

