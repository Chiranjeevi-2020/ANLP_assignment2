"""
Decoder-only Transformer implementation with drop-in MoE support.
Follows Appendix Listing 1 and requirements of Assignment 2.
"""

import math
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig
from .moe import MoE, MLP


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism utilizing scaled_dot_product_attention.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.D_model % config.N_heads == 0
        self.D_model = config.D_model
        self.N_heads = config.N_heads
        self.head_dim = config.D_model // config.N_heads
        self.dropout = config.dropout

        # Combined Q, K, V projection
        self.c_attn = nn.Linear(config.D_model, 3 * config.D_model, bias=True)
        # Output projection
        self.c_proj = nn.Linear(config.D_model, config.D_model, bias=True)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch size, sequence length, hidden dimension

        # Compute queries, keys, and values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.D_model, dim=2)

        # Reshape for multi-head attention: (B, N_heads, T, head_dim)
        q = q.view(B, T, self.N_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.N_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.N_heads, self.head_dim).transpose(1, 2)

        # PyTorch 2.0+ scaled_dot_product_attention with causal masking
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        # Re-assemble head outputs: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(y))
        return out


class TransformerBlock(nn.Module):
    """
    Standard Transformer Decoder Block containing:
    - LayerNorm + Causal Attention + Residual
    - LayerNorm + FFN (MLP or MoE) + Residual
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.D_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.D_model)
        self.ffn = MoE(config) if config.use_moe else MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Decoder-only Transformer Language Model.
    Supports drop-in MoE or standard MLP feed-forward networks.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.N_vocab, config.D_model)
        self.wpe = nn.Embedding(config.N_ctx, config.D_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.N_layers)
        ])
        self.ln_f = nn.LayerNorm(config.D_model)
        self.head = nn.Linear(config.D_model, config.N_vocab, bias=False)

        # Tie weights between token embeddings and output projection
        self.head.weight = self.wte.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_aux_loss(self) -> torch.Tensor:
        """Sums auxiliary load-balancing losses across all MoE layers."""
        if not self.config.use_moe:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total_aux = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            if isinstance(block.ffn, MoE):
                total_aux = total_aux + block.ffn.last_aux_loss.to(total_aux.device)
        return total_aux

    def get_active_parameters(self) -> int:
        """
        Computes the number of parameters actively participating in forward computation per token.
        For dense models, active_parameters == total_parameters.
        For MoE models, only router and active experts count.
        """
        active = 0
        # Embeddings
        active += self.wte.weight.numel()
        active += self.wpe.weight.numel()
        # Final norm and tied lm_head
        active += sum(p.numel() for p in self.ln_f.parameters())
        # (self.head shares weights with self.wte, so not double-counted)

        for block in self.blocks:
            active += sum(p.numel() for p in block.ln_1.parameters())
            active += sum(p.numel() for p in block.attn.parameters())
            active += sum(p.numel() for p in block.ln_2.parameters())
            if isinstance(block.ffn, MoE):
                active += block.ffn.get_active_parameters()
            else:
                active += sum(p.numel() for p in block.ffn.parameters())

        return active

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass for Transformer.
        Args:
            idx: Tensor of token indices with shape (B, T)
            targets: Optional ground-truth target tokens with shape (B, T)
        Returns:
            logits: (B, T, N_vocab)
            loss: Cross-entropy loss + aux_loss (if targets provided)
            aux_loss: Load balancing auxiliary loss
        """
        device = idx.device
        B, T = idx.shape
        assert T <= self.config.N_ctx, f"Sequence length {T} exceeds context window {self.config.N_ctx}"

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        # Token + positional embeddings
        tok_emb = self.wte(idx)       # (B, T, D_model)
        pos_emb = self.wpe(pos)       # (1, T, D_model)
        x = self.drop(tok_emb + pos_emb)

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, N_vocab)

        loss = None
        aux_loss = self.get_aux_loss()

        if targets is not None:
            # Shift tokens for next-token prediction
            if targets.shape[1] == logits.shape[1]:
                # Standard next-token prediction: position t predicts token t+1
                shift_logits = logits[:, :-1, :].contiguous().view(-1, self.config.N_vocab)
                shift_targets = targets[:, 1:].contiguous().view(-1)
            elif targets.shape[1] == logits.shape[1] - 1:
                # Targets are already pre-shifted by 1 token
                shift_logits = logits[:, :-1, :].contiguous().view(-1, self.config.N_vocab)
                shift_targets = targets.contiguous().view(-1)
            else:
                shift_logits = logits.contiguous().view(-1, self.config.N_vocab)
                shift_targets = targets.contiguous().view(-1)

            ce_loss = F.cross_entropy(shift_logits, shift_targets, ignore_index=-100)
            loss = ce_loss + aux_loss

        return logits, loss, aux_loss
