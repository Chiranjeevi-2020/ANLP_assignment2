"""
ANLP Assignment 2: Utilities Module
Contains helper functions for:
- Random seed reproduction
- Parameter counting (total vs active parameters for MoE)
- Metric computations (Perplexity, BLEU, ROUGE)
- WandB / Local experiment logging
- Benchmarking and timing
"""

import os
import math
import time
import random
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("ANLP_Utils")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Computes both total parameters and active parameters per token.
    For standard dense models, active_parameters == total_parameters.
    For MoE models, active_parameters accounts for only the router + active experts per token.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Check if model has custom get_active_parameters method
    if hasattr(model, "get_active_parameters"):
        active_params = model.get_active_parameters()
    else:
        active_params = total_params

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "active_parameters": active_params,
    }


def compute_perplexity(loss: float) -> float:
    """Computes perplexity from cross-entropy loss with numerical clamping."""
    try:
        if math.isnan(loss) or math.isinf(loss) or loss > 100.0:
            return float("inf")
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
    smooth_method: str = "floor"
) -> float:
    """
    Computes corpus-level BLEU-4 score.
    Attempts to use sacrebleu or nltk; provides built-in fallback if unavailable.
    """
    # Clean inputs
    preds = [p.strip() for p in predictions]
    refs = [[r.strip() for r in ref_list] for ref_list in references]

    try:
        import sacrebleu
        # sacrebleu expects references as list of lists across each ref position
        # e.g., [[ref1_doc1, ref1_doc2], [ref2_doc1, ref2_doc2]]
        num_refs = max(len(r) for r in refs)
        transposed_refs = []
        for i in range(num_refs):
            transposed_refs.append([r[i] if i < len(r) else r[0] for r in refs])
        bleu = sacrebleu.corpus_bleu(preds, transposed_refs)
        return float(bleu.score)
    except ImportError:
        pass

    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1 if smooth_method == "floor" else SmoothingFunction().method2
        tokenized_preds = [p.split() for p in preds]
        tokenized_refs = [[r.split() for r in ref_list] for ref_list in refs]
        score = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smooth)
        return float(score * 100.0)
    except ImportError:
        pass

    # Simple built-in n-gram BLEU fallback (unigram/bigram overlap)
    total_matches = 0
    total_possible = 0
    for p, r_list in zip(preds, refs):
        p_tokens = p.lower().split()
        if not p_tokens:
            continue
        ref_tokens = r_list[0].lower().split() if r_list else []
        matches = sum(1 for tok in p_tokens if tok in ref_tokens)
        total_matches += matches
        total_possible += len(p_tokens)
    if total_possible == 0:
        return 0.0
    return float((total_matches / total_possible) * 100.0)


class WandbLogger:
    """
    Robust logger wrapping Weights & Biases with safe local fallback.
    Prevents pipeline failures if WandB API key is missing or offline.
    """
    def __init__(
        self,
        project: str = "ANLP-Assignment-2",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.run = None
        self.local_history: List[Dict[str, Any]] = []

        if self.enabled:
            try:
                import wandb
                # If WANDB_API_KEY is not set in environment, wandb can run in offline mode
                mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
                self.run = wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    tags=tags,
                    mode=mode,
                    reinit=True
                )
                logger.info(f"WandB initialized successfully (mode={mode}, run={name})")
            except Exception as e:
                logger.warning(f"Could not initialize WandB ({e}). Defaulting to local logging.")
                self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Logs metrics to WandB and stores in local history."""
        entry = dict(metrics)
        if step is not None:
            entry["step"] = step
        entry["timestamp"] = time.time()
        self.local_history.append(entry)

        if self.enabled and self.run is not None:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to push metrics to WandB: {e}")

    def finish(self) -> None:
        """Closes WandB run."""
        if self.enabled and self.run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    def save_local_history(self, filepath: str) -> None:
        """Saves local metric logs to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.local_history, f, indent=2)
        logger.info(f"Saved local run log to {filepath}")
