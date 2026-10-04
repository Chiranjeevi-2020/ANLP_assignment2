"""
Configuration dataclass for Transformer and Mixture-of-Experts (MoE) architectures.
Matches Listing 1 from Appendix with full parameter-accounting formulas for the 5 ablations.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class TransformerConfig:
    # Model architecture dimensions
    D_model: int = 256                  # Hidden representation dimension
    N_ctx: int = 256                    # Maximum context window length
    N_vocab: int = 16000                # Vocabulary size
    N_layers: int = 6                   # Number of Transformer blocks
    N_heads: int = 8                    # Multi-head attention heads
    D_ff: int = 1024                    # Feed-forward hidden dimension (4 * D_model)
    dropout: float = 0.1                # Dropout probability

    # MoE architecture configuration
    use_moe: bool = False               # False: Standard MLP; True: MoE
    ablation_variant: Literal[
        "dense",
        "top1",
        "top2",
        "shared_routed",
        "matched_active"
    ] = "dense"

    # MoE details
    num_experts: int = 4                # Total experts in MoE layer
    num_active_experts: int = 1         # Active experts routed per token
    num_shared_experts: int = 0         # Always-active shared experts (for variant 4)
    aux_loss_coef: float = 0.01         # Switch/GShard load balancing auxiliary loss weight
    capacity_factor: float = 1.25       # Capacity factor for expert routing if capped

    # Training settings
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 10000
    batch_size: int = 32

    def __post_init__(self):
        """Validates configuration parameters and assigns ablation-specific settings."""
        assert self.D_model % self.N_heads == 0, f"D_model ({self.D_model}) must be divisible by N_heads ({self.N_heads})"

        if self.ablation_variant == "dense":
            self.use_moe = False
        else:
            self.use_moe = True

        if self.ablation_variant == "top1":
            # 4 experts, 1 active at a time. Total params == Dense baseline.
            self.num_experts = 4
            self.num_active_experts = 1
            self.num_shared_experts = 0
            self.expert_d_ff = self.D_ff // 4
        elif self.ablation_variant == "top2":
            # 4 experts, 2 active at a time. Total params == Dense baseline.
            self.num_experts = 4
            self.num_active_experts = 2
            self.num_shared_experts = 0
            self.expert_d_ff = self.D_ff // 4
        elif self.ablation_variant == "shared_routed":
            # 1 shared expert (always active) + 3 routed (1 active). Total 2 of 4 active.
            # Total params == Dense baseline.
            self.num_experts = 4
            self.num_shared_experts = 1
            self.num_active_experts = 1  # 1 routed active + 1 shared active = 2 active total
            self.expert_d_ff = self.D_ff // 4
        elif self.ablation_variant == "matched_active":
            # 4 experts, 2 active at a time, ACTIVE params == Dense baseline.
            # Since 2 are active, each expert has D_ff / 2, so active = 2 * (D_ff / 2) = D_ff.
            # Total params = 4 * (D_ff / 2) = 2 * Dense.
            self.num_experts = 4
            self.num_active_experts = 2
            self.num_shared_experts = 0
            self.expert_d_ff = self.D_ff // 2
        else:
            self.expert_d_ff = self.D_ff

    @property
    def ffn_total_params_per_layer(self) -> int:
        """Computes the theoretical total parameter count of one FFN/MoE layer."""
        if not self.use_moe or self.ablation_variant == "dense":
            # Linear1: D_model * D_ff + D_ff; Linear2: D_ff * D_model + D_model
            return 2 * self.D_model * self.D_ff + self.D_ff + self.D_model
        elif self.ablation_variant in ["top1", "top2"]:
            # Router: D_model * 4 + 4 experts of (2 * D_model * d_expert)
            router = self.D_model * self.num_experts
            experts = self.num_experts * (2 * self.D_model * self.expert_d_ff + self.expert_d_ff + self.D_model)
            return router + experts
        elif self.ablation_variant == "shared_routed":
            # Router: D_model * 3 + 1 shared expert + 3 routed experts
            router = self.D_model * 3
            shared = 1 * (2 * self.D_model * self.expert_d_ff + self.expert_d_ff + self.D_model)
            routed = 3 * (2 * self.D_model * self.expert_d_ff + self.expert_d_ff + self.D_model)
            return router + shared + routed
        elif self.ablation_variant == "matched_active":
            router = self.D_model * self.num_experts
            experts = self.num_experts * (2 * self.D_model * self.expert_d_ff + self.expert_d_ff + self.D_model)
            return router + experts
        return 0

    @property
    def ffn_active_params_per_layer(self) -> int:
        """Computes the theoretical active parameter count of one FFN/MoE layer per token."""
        if not self.use_moe or self.ablation_variant == "dense":
            return self.ffn_total_params_per_layer
        elif self.ablation_variant == "top1":
            router = self.D_model * self.num_experts
            active_expert = 1 * (2 * self.D_model * self.expert_d_ff + self.expert_d_ff + self.D_model)
            return router + active_expert
        elif self.ablation_variant == "top2":
            router = self.D_model * self.num_experts
            active_experts = 2 * (2 * self.D_model * self.expert_d_ff + self.expert_d_ff + self.D_model)
            return router + active_experts
        elif self.ablation_variant == "shared_routed":
            router = self.D_model * 3
            shared_expert = 1 * (2 * self.D_model * self.expert_d_ff + self.expert_d_ff + self.D_model)
            active_routed = 1 * (2 * self.D_model * self.expert_d_ff + self.expert_d_ff + self.D_model)
            return router + shared_expert + active_routed
        elif self.ablation_variant == "matched_active":
            router = self.D_model * self.num_experts
            active_experts = 2 * (2 * self.D_model * self.expert_d_ff + self.expert_d_ff + self.D_model)
            return router + active_experts
        return 0
