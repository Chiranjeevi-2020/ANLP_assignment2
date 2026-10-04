"""
Mixture of Experts (MoE) implementation for Transformer architectures.
Implements the 5 required FFN variants:
1. Standard 2-layer MLP (dense baseline)
2. MoE with 4 experts, 1 active at a time (top-1 routing)
3. MoE with 4 experts, 2 active at a time (top-2 routing)
4. MoE with 1 shared expert (always active) + 3 routed experts (1 active at a time, 2 total active)
5. MoE with 4 experts, 2 active at a time, matched active parameters to dense baseline (1)
"""

import math
from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig


class MLP(nn.Module):
    """
    Standard 2-layer Feed-Forward Network (Dense Baseline).
    Input:  (B, N_ctx, D_model)
    Output: (B, N_ctx, D_model)
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.D_model = config.D_model
        self.D_ff = config.D_ff
        self.w1 = nn.Linear(config.D_model, config.D_ff, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.w2 = nn.Linear(config.D_ff, config.D_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_ctx, D_model)
        h = self.act(self.w1(x))
        h = self.dropout(h)
        out = self.dropout(self.w2(h))
        return out


class Expert(nn.Module):
    """
    Individual Expert Feed-Forward Network.
    Input:  (..., D_model)
    Output: (..., D_model)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.w1(x))
        h = self.dropout(h)
        return self.dropout(self.w2(h))


class MoE(nn.Module):
    """
    Mixture-of-Experts Module with drop-in replacement support for MLP.
    Supports Top-1, Top-2, Shared+Routed, and Matched-Active configurations.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.D_model = config.D_model
        self.ablation_variant = config.ablation_variant
        self.aux_loss_coef = config.aux_loss_coef

        # Determine architecture based on ablation variant
        if self.ablation_variant == "shared_routed":
            # 1 shared expert + 3 routed experts
            self.num_routed_experts = 3
            self.shared_expert = Expert(config.D_model, config.expert_d_ff, config.dropout)
            self.routed_experts = nn.ModuleList([
                Expert(config.D_model, config.expert_d_ff, config.dropout)
                for _ in range(self.num_routed_experts)
            ])
            self.router = nn.Linear(config.D_model, self.num_routed_experts, bias=False)
            self.top_k = 1
        else:
            # top1, top2, matched_active
            self.num_routed_experts = config.num_experts
            self.shared_expert = None
            self.routed_experts = nn.ModuleList([
                Expert(config.D_model, config.expert_d_ff, config.dropout)
                for _ in range(self.num_routed_experts)
            ])
            self.router = nn.Linear(config.D_model, self.num_routed_experts, bias=False)
            self.top_k = config.num_active_experts

        # Running telemetry for expert utilization and language analysis
        self.register_buffer(
            "expert_selection_counts",
            torch.zeros(self.num_routed_experts, dtype=torch.int64),
            persistent=False
        )
        self.last_expert_indices: Optional[torch.Tensor] = None
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)

    def get_active_parameters(self) -> int:
        """Computes active parameters per token for this layer."""
        router_params = sum(p.numel() for p in self.router.parameters())
        if self.ablation_variant == "shared_routed":
            shared_params = sum(p.numel() for p in self.shared_expert.parameters())
            routed_params = sum(p.numel() for p in self.routed_experts[0].parameters()) * self.top_k
            return router_params + shared_params + routed_params
        else:
            routed_params = sum(p.numel() for p in self.routed_experts[0].parameters()) * self.top_k
            return router_params + routed_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MoE layer.
        Args:
            x: Input tensor of shape (B, N_ctx, D_model)
        Returns:
            out: Output tensor of shape (B, N_ctx, D_model)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)  # (N, D) where N = B * N_ctx
        num_tokens = x_flat.size(0)

        # 1. Router forward
        router_logits = self.router(x_flat)  # (N, num_routed_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, num_routed_experts)

        # 2. Expert selection
        if self.top_k == 1:
            top_weights, top_indices = router_probs.max(dim=-1, keepdim=True)  # (N, 1)
        else:
            top_weights, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)  # (N, top_k)
            # Re-normalize weights among selected top-k experts
            top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Cache routing decisions for downstream language heatmap extraction
        self.last_expert_indices = top_indices.detach()

        # Update cumulative selection counts
        with torch.no_grad():
            for i in range(self.num_routed_experts):
                self.expert_selection_counts[i] += (top_indices == i).sum().item()

        # 3. Compute auxiliary load-balancing loss (Switch / GShard formulation)
        # L_aux = alpha * num_experts * sum_i(f_i * P_i)
        # where f_i is fraction of tokens dispatched to expert i, P_i is average routing prob
        num_experts = self.num_routed_experts
        expert_mask = F.one_hot(top_indices, num_classes=num_experts)  # (N, top_k, num_experts)
        tokens_per_expert = expert_mask.sum(dim=1).sum(dim=0).float()  # (num_experts,)
        f_i = tokens_per_expert / (num_tokens * self.top_k + 1e-9)
        P_i = router_probs.mean(dim=0)  # (num_experts,)
        self.last_aux_loss = self.aux_loss_coef * (num_experts * torch.sum(f_i * P_i))

        # 4. Dispatch tokens through routed experts
        routed_out = torch.zeros_like(x_flat)
        for expert_idx in range(num_experts):
            # Check which tokens selected this expert
            # top_indices shape: (N, top_k)
            token_matches = (top_indices == expert_idx)
            if not token_matches.any():
                continue

            token_indices, k_positions = torch.where(token_matches)
            selected_tokens = x_flat[token_indices]  # (M, D)
            weights = top_weights[token_indices, k_positions].unsqueeze(-1)  # (M, 1)

            expert_output = self.routed_experts[expert_idx](selected_tokens)
            routed_out.index_add_(0, token_indices, expert_output * weights)

        # 5. Add shared expert output if applicable
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)
            total_out = shared_out + routed_out
        else:
            total_out = routed_out

        return total_out.reshape(batch_size, seq_len, d_model)
