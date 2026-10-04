"""
Part 1: Mixture of Experts (MoE) Module
Drop-in replacement for transformer feed-forward networks with 5 architectural ablations.
"""

from .config import TransformerConfig
from .moe import MoE, MLP, Expert
from .model import Transformer, CausalSelfAttention, TransformerBlock
from .dataset import build_or_load_tokenizer, TranslationDataset, collate_fn
from .train import evaluate_and_extract_heatmaps, save_expert_heatmap

__all__ = [
    "TransformerConfig",
    "MoE",
    "MLP",
    "Expert",
    "Transformer",
    "CausalSelfAttention",
    "TransformerBlock",
    "build_or_load_tokenizer",
    "TranslationDataset",
    "collate_fn",
    "evaluate_and_extract_heatmaps",
    "save_expert_heatmap",
]
