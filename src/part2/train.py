"""
Part 2 Pretraining script comparing Custom Optimizers.
Evaluates validation loss and test BLEU at 0.1x dataset token checkpoints.
Produces Section 4 / Figure 2 training dynamic plots.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..part1.config import TransformerConfig
from ..part1.model import Transformer
from .optimizers import CustomAdamW, CustomAdan, CustomLion, CustomMuon, CustomSophia
from ..utils import set_seed, compute_perplexity, compute_bleu, WandbLogger

logger = logging.getLogger("Part2_Train")


def get_optimizer(
    optimizer_name: str,
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """Instantiates custom optimizer without using any predefined torch.optim modules."""
    name = optimizer_name.lower()
    if name == "adamw":
        return CustomAdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adan":
        return CustomAdan(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "lion":
        # Lion typically operates with 3x-10x lower lr and higher weight decay
        return CustomLion(model.parameters(), lr=lr * 0.2, weight_decay=weight_decay * 5.0)
    elif name == "muon":
        return CustomMuon(model.parameters(), lr=lr * 2.0, fallback_lr=lr, weight_decay=weight_decay)
    elif name == "sophia":
        return CustomSophia(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


@torch.no_grad()
def evaluate_val_and_bleu(
    model: Transformer,
    val_loader: DataLoader,
    eval_pairs: List[Tuple[str, str]],
    tokenizer: Any,
    device: torch.device,
    max_eval_pairs: int = 200
) -> Tuple[float, float, float]:
    """Computes validation loss, validation perplexity, and test BLEU score."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        logits, loss, _ = model(input_ids, targets=targets)
        num_tok = (targets != -100).sum().item()
        total_loss += loss.item() * num_tok
        total_tokens += max(num_tok, 1)

    val_loss = total_loss / max(total_tokens, 1)
    val_ppl = compute_perplexity(val_loss)

    # Compute BLEU on prefix-continuation test pairs
    preds, refs = [], []
    for prompt, gold_continuation in eval_pairs[:max_eval_pairs]:
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        if prompt_ids.shape[1] >= model.config.N_ctx:
            continue

        curr_ids = prompt_ids.clone()
        for _ in range(30):
            if curr_ids.shape[1] >= model.config.N_ctx:
                break
            logits, _, _ = model(curr_ids)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_tok], dim=1)
            if next_tok.item() == tokenizer.eos_token_id:
                break

        gen_tokens = curr_ids[0, prompt_ids.shape[1]:]
        pred_text = tokenizer.decode(gen_tokens.cpu(), skip_special_tokens=True)
        preds.append(pred_text)
        refs.append([gold_continuation])

    test_bleu = compute_bleu(preds, refs)
    return val_loss, val_ppl, test_bleu


def run_optimizer_benchmark(
    optimizer_names: List[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_pairs: List[Tuple[str, str]],
    tokenizer: Any,
    config: TransformerConfig,
    total_budget_tokens: int,
    device: torch.device,
    checkpoints_count: int = 10,
    output_dir: str = "assets/part2"
) -> Dict[str, Dict[str, List[float]]]:
    """
    Trains models with different optimizers up to 1.0x total dataset tokens,
    recording val loss and test BLEU at 0.1x dataset intervals (Chinchilla curve).
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_interval_tokens = total_budget_tokens // checkpoints_count
    all_results: Dict[str, Dict[str, List[float]]] = {}

    for opt_name in optimizer_names:
        logger.info(f"--- Starting training run for Optimizer: {opt_name.upper()} ---")
        set_seed(42)
        model = Transformer(config).to(device)
        optimizer = get_optimizer(opt_name, model, lr=config.learning_rate, weight_decay=config.weight_decay)

        wandb_logger = None
        if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
            try:
                wandb_logger = WandbLogger(
                    project="ANLP-Assignment-2",
                    name=f"Part2-Opt-{opt_name}",
                    config={"optimizer": opt_name, "budget_tokens": total_budget_tokens}
                )
            except Exception as e:
                logger.warning(f"Could not initialize WandB ({e}).")

        tokens_seen = 0
        next_checkpoint_tokens = checkpoint_interval_tokens
        history: Dict[str, List[float]] = {
            "tokens": [],
            "val_loss": [],
            "val_ppl": [],
            "test_bleu": []
        }

        # Initial zero-shot evaluation
        v_loss, v_ppl, t_bleu = evaluate_val_and_bleu(model, val_loader, eval_pairs, tokenizer, device)
        history["tokens"].append(0)
        history["val_loss"].append(v_loss)
        history["val_ppl"].append(v_ppl)
        history["test_bleu"].append(t_bleu)
        if wandb_logger:
            wandb_logger.log({
                "tokens_seen": 0,
                "val/loss": v_loss,
                "val/perplexity": v_ppl,
                "test/bleu": t_bleu
            }, step=0)

        model.train()
        train_iter = iter(train_loader)

        while tokens_seen < total_budget_tokens:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            logits, loss, _ = model(input_ids, targets=targets)
            loss.backward()

            # Sophia requires occasional hessian update
            if opt_name == "sophia" and hasattr(optimizer, "update_hessian"):
                optimizer.update_hessian()

            optimizer.step()

            batch_tokens = input_ids.numel()
            tokens_seen += batch_tokens

            if tokens_seen >= next_checkpoint_tokens:
                v_loss, v_ppl, t_bleu = evaluate_val_and_bleu(model, val_loader, eval_pairs, tokenizer, device)
                history["tokens"].append(tokens_seen)
                history["val_loss"].append(v_loss)
                history["val_ppl"].append(v_ppl)
                history["test_bleu"].append(t_bleu)
                if wandb_logger:
                    wandb_logger.log({
                        "tokens_seen": tokens_seen,
                        "val/loss": v_loss,
                        "val/perplexity": v_ppl,
                        "test/bleu": t_bleu
                    }, step=tokens_seen)
                logger.info(f"[{opt_name}] Tokens: {tokens_seen}/{total_budget_tokens} | Val Loss: {v_loss:.4f} | BLEU: {t_bleu:.2f}")
                next_checkpoint_tokens += checkpoint_interval_tokens
                model.train()

        if wandb_logger:
            wandb_logger.finish()

        all_results[opt_name] = history

    # Save comparative plots
    plot_comparative_dynamics(all_results, save_path=os.path.join(output_dir, "optimizer_dynamics.png"))

    with open(os.path.join(output_dir, "optimizer_benchmark_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def plot_comparative_dynamics(
    results: Dict[str, Dict[str, List[float]]],
    save_path: str = "assets/optimizer_dynamics.png"
) -> None:
    """Plots Figure 2 equivalent: Validation Loss and Test BLEU vs Tokens."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for opt_name, history in results.items():
        tokens = [t / 1e6 for t in history["tokens"]]  # In millions of tokens
        ax1.plot(tokens, history["val_loss"], marker="o", label=opt_name.upper())
        ax2.plot(tokens, history["test_bleu"], marker="s", label=opt_name.upper())

    ax1.set_title("Validation Loss vs Training Tokens (Figure 2 Replication)")
    ax1.set_xlabel("Tokens (Millions)")
    ax1.set_ylabel("Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_title("Test BLEU Score vs Training Tokens")
    ax2.set_xlabel("Tokens (Millions)")
    ax2.set_ylabel("BLEU Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved optimizer dynamics plot to {save_path}")


def run_part2_pipeline(
    optimizer_names: Optional[List[str]] = None,
    dataset_name: str = "browndw/human-ai-parallel-corpus",
    total_budget_tokens: int = 200_000,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    output_dir: str = "assets/part2"
) -> Dict[str, Any]:
    """
    Turnkey pipeline for benchmarking custom optimizers on browndw/human-ai-parallel-corpus.
    Loads data, trains models, logs intermediate 0.1x token checkpoints, and plots Chinchilla curves.
    """
    from .dataset import PretrainDataset
    from ..part1.dataset import build_or_load_tokenizer

    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if optimizer_names is None:
        optimizer_names = ["adamw", "adan", "lion", "muon", "sophia"]

    config = TransformerConfig(ablation_variant="dense", D_model=256, N_ctx=128, D_ff=1024, N_layers=4)
    tokenizer = build_or_load_tokenizer(vocab_size=config.N_vocab)

    logger.info(f"Loading pretraining corpus: {dataset_name}...")
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        texts = []
        for i, item in enumerate(ds):
            if i >= 5000:
                break
            # Collect text fields
            t = item.get("text_human") or item.get("text") or item.get("text_ai") or ""
            if t:
                texts.append(t)
    except Exception as e:
        logger.warning(f"Could not load Hugging Face dataset ({e}). Using synthetic demonstration corpus.")
        texts = [
            "Language models learn probabilistic distributions over sequence tokens through self-supervised pretraining.",
            "Optimization trajectory depends heavily on learning rate schedules, momentum damping, and gradient geometry.",
            "Second order and matrix based optimizers like Muon and Sophia provide faster loss reduction per token."
        ] * 300

    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:] if split_idx < len(texts) else texts[:10]

    train_ds = PretrainDataset(train_texts, tokenizer, seq_len=config.N_ctx)
    val_ds = PretrainDataset(val_texts, tokenizer, seq_len=config.N_ctx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    eval_pairs = val_ds.eval_pairs if len(val_ds.eval_pairs) > 0 else [
        ("Language models learn", "probabilistic distributions over sequence tokens."),
        ("Optimization trajectory depends", "heavily on learning rate schedules.")
    ]

    results = run_optimizer_benchmark(
        optimizer_names=optimizer_names,
        train_loader=train_loader,
        val_loader=val_loader,
        eval_pairs=eval_pairs,
        tokenizer=tokenizer,
        config=config,
        total_budget_tokens=total_budget_tokens,
        device=device,
        output_dir=output_dir
    )

    # Also copy plot to assets root
    root_plot = "assets/optimizer_dynamics.png"
    plot_comparative_dynamics(results, save_path=root_plot)

    return results
