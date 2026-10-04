"""
Training and evaluation pipeline for Part 1 (MoE Transformer Ablations).
Evaluates perplexity, BLEU, expert utilization heatmaps, and saves artifacts.
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
import numpy as np

from .config import TransformerConfig
from .model import Transformer
from .moe import MoE
from .dataset import build_or_load_tokenizer, TranslationDataset, collate_fn
from ..utils import set_seed, compute_perplexity, compute_bleu, WandbLogger, count_parameters

logger = logging.getLogger("Part1_Train")


def train_epoch(
    model: Transformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    wandb_logger: Optional[WandbLogger] = None,
    step_offset: int = 0
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    step = step_offset

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        logits, loss, aux_loss = model(input_ids, targets=targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        num_valid_tokens = (targets != -100).sum().item()
        total_loss += loss.item() * num_valid_tokens
        total_tokens += max(num_valid_tokens, 1)

        if wandb_logger and step % 20 == 0:
            wandb_logger.log({
                "train/loss": loss.item(),
                "train/aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
                "train/step": step
            }, step=step)

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, step


@torch.no_grad()
def evaluate_and_extract_heatmaps(
    model: Transformer,
    dataloader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    max_eval_samples: int = 1000
) -> Dict[str, Any]:
    """
    Evaluates perplexity and BLEU, while recording layer-wise expert activations
    split by input language ('vi' vs 'ja').
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    predictions: List[str] = []
    references: List[List[str]] = []

    # Track expert activation counts: lang -> layer_idx -> [count_per_expert]
    has_moe = model.config.use_moe
    num_layers = len(model.blocks)
    routed_experts_count = (
        model.config.num_experts if model.config.ablation_variant != "shared_routed" else 3
    )

    expert_activity = {
        "vi": np.zeros((num_layers, routed_experts_count), dtype=np.float64),
        "ja": np.zeros((num_layers, routed_experts_count), dtype=np.float64)
    }
    lang_token_counts = {"vi": 0, "ja": 0}

    eval_samples_processed = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        src_langs = batch["src_langs"]
        prompts = batch["prompts"]
        tgt_texts = batch["tgt_texts"]

        logits, loss, _ = model(input_ids, targets=targets)
        num_valid_tokens = (targets != -100).sum().item()
        total_loss += loss.item() * num_valid_tokens
        total_tokens += max(num_valid_tokens, 1)

        # Record expert routing decisions if MoE
        if has_moe:
            for l_idx, block in enumerate(model.blocks):
                if isinstance(block.ffn, MoE) and block.ffn.last_expert_indices is not None:
                    # last_expert_indices: (B * N_ctx, top_k)
                    indices = block.ffn.last_expert_indices.view(input_ids.shape[0], input_ids.shape[1], -1)
                    for b_idx, lang in enumerate(src_langs):
                        if lang in expert_activity:
                            sample_indices = indices[b_idx].cpu().numpy().flatten()
                            for exp_id in sample_indices:
                                if 0 <= exp_id < routed_experts_count:
                                    expert_activity[lang][l_idx, exp_id] += 1
                                    lang_token_counts[lang] += 1

        # Greedy decoding for BLEU evaluation on a subset of samples
        if eval_samples_processed < max_eval_samples:
            for b_idx in range(len(prompts)):
                prompt = prompts[b_idx]
                target_text = tgt_texts[b_idx]
                prompt_enc = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                # Generate up to 50 tokens
                curr_ids = prompt_enc.clone()
                for _ in range(50):
                    if curr_ids.shape[1] >= model.config.N_ctx:
                        break
                    out_logits, _, _ = model(curr_ids)
                    next_tok = out_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    curr_ids = torch.cat([curr_ids, next_tok], dim=1)
                    if next_tok.item() == tokenizer.eos_token_id:
                        break

                gen_tokens = curr_ids[0, prompt_enc.shape[1]:]
                pred_text = tokenizer.decode(gen_tokens.cpu(), skip_special_tokens=True)
                predictions.append(pred_text)
                references.append([target_text])

                eval_samples_processed += 1
                if eval_samples_processed >= max_eval_samples:
                    break

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = compute_perplexity(avg_loss)
    bleu = compute_bleu(predictions, references)

    # Normalize expert activity to distribution per language
    for lang in ["vi", "ja"]:
        total_act = expert_activity[lang].sum()
        if total_act > 0:
            expert_activity[lang] = expert_activity[lang] / total_act

    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "bleu": bleu,
        "expert_activity": expert_activity,
        "sample_predictions": predictions[:5],
        "sample_references": references[:5]
    }


def save_expert_heatmap(
    expert_activity: Dict[str, np.ndarray],
    save_path: str = "assets/expert_heatmap.png",
    variant_name: str = "MoE"
) -> None:
    """Plots and saves expert activation heatmap comparing Vietnamese vs Japanese."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, lang in enumerate(["vi", "ja"]):
        ax = axes[idx]
        data = expert_activity[lang]
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=float)
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_title(f"Expert Activation: {lang.upper()} ({variant_name})")
        ax.set_xlabel("Expert Index")
        ax.set_ylabel("Transformer Layer")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Expert activation heatmap saved to {save_path}")


def run_part1_training(
    ablation_variant: str = "dense",
    config: Optional[TransformerConfig] = None,
    dataset_name: str = "belumind/en-vi-ja-curated-500k-triplets",
    max_samples: Optional[int] = 10000,
    batch_size: int = 32,
    epochs: int = 1,
    max_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    output_dir: str = "assets/part1",
    save_checkpoint_dir: str = "checkpoints/part1"
) -> Dict[str, Any]:
    """
    End-to-end training and evaluation pipeline for Part 1 MoE ablations.
    Trains model, extracts expert language specialization, and saves heatmaps.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config is None:
        config = TransformerConfig(ablation_variant=ablation_variant)

    logger.info(f"Initializing model for ablation: {ablation_variant.upper()} on device {device}")
    model = Transformer(config).to(device)

    # Load or build multilingual byte-level BPE tokenizer
    tokenizer = build_or_load_tokenizer(vocab_size=config.N_vocab)

    # Load translation dataset
    logger.info(f"Loading dataset {dataset_name} (max_samples={max_samples})...")
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        raw_samples = []
        for i, item in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            raw_samples.append(item)
    except Exception as e:
        logger.warning(f"Could not load Hugging Face dataset ({e}). Using demonstration seed corpus.")
        raw_samples = [
            {"en": "Hello, how are you today?", "vi": "Xin chào, hôm nay bạn thế nào?", "ja": "こんにちは、今日はお元気ですか？"},
            {"en": "Machine translation with neural networks is fascinating.", "vi": "Dịch máy với mạng nơ-ron rất hấp dẫn.", "ja": "ニューラルネットワークによる機械翻訳は非常に興味深い。"},
            {"en": "Mixture of experts scales model capacity efficiently.", "vi": "Mô hình chuyên gia kết hợp mở rộng dung lượng hiệu quả.", "ja": "エキスパートの混合はモデル容量を効率的に拡張します。"},
            {"en": "Artificial intelligence continues to transform language technology.", "vi": "Trí tuệ nhân tạo tiếp tục chuyển đổi công nghệ ngôn ngữ.", "ja": "人工知能は言語技術を変革し続けています。"}
        ] * 200

    # Split train and validation
    split_idx = int(len(raw_samples) * 0.9)
    train_samples = raw_samples[:split_idx]
    val_samples = raw_samples[split_idx:] if split_idx < len(raw_samples) else raw_samples[:10]

    train_ds = TranslationDataset(train_samples, tokenizer, max_seq_len=config.N_ctx)
    val_ds = TranslationDataset(val_samples, tokenizer, max_seq_len=config.N_ctx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    wandb_logger = None
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
        try:
            wandb_logger = WandbLogger(
                project="ANLP-Assignment-2",
                name=f"Part1-MoE-{ablation_variant}",
                config={"ablation": ablation_variant, "epochs": epochs, "samples": max_samples}
            )
        except Exception as e:
            logger.warning(f"Could not initialize WandB ({e}).")

    logger.info(f"Starting training for {epochs} epoch(s)...")
    step_offset = 0
    for epoch in range(epochs):
        avg_loss, step_offset = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            wandb_logger=wandb_logger,
            step_offset=step_offset
        )
        logger.info(f"Epoch {epoch + 1}/{epochs} | Average Training Loss: {avg_loss:.4f}")

    # Evaluate and extract expert heatmaps
    logger.info("Evaluating model and extracting language routing decisions...")
    eval_results = evaluate_and_extract_heatmaps(model, val_loader, tokenizer, device)

    if wandb_logger:
        wandb_logger.log({
            "val/loss": eval_results["loss"],
            "val/perplexity": eval_results["perplexity"],
            "val/bleu": eval_results["bleu"]
        })
        wandb_logger.finish()

    # Save heatmap if MoE
    if config.use_moe:
        heatmap_path = os.path.join(output_dir, f"expert_heatmap_{ablation_variant}.png")
        save_expert_heatmap(eval_results["expert_activity"], save_path=heatmap_path, variant_name=ablation_variant)
        # Also copy to root assets
        save_expert_heatmap(eval_results["expert_activity"], save_path="assets/expert_heatmap.png", variant_name=ablation_variant)

    # Save checkpoint
    ckpt_path = os.path.join(save_checkpoint_dir, f"model_{ablation_variant}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "eval_results": {k: v for k, v in eval_results.items() if k != "expert_activity"}
    }, ckpt_path)
    logger.info(f"Checkpoint saved to {ckpt_path}")

    # Save evaluation summary
    summary_path = os.path.join(output_dir, f"summary_{ablation_variant}.json")
    with open(summary_path, "w") as f:
        json.dump({
            "variant": ablation_variant,
            "loss": eval_results["loss"],
            "perplexity": eval_results["perplexity"],
            "bleu": eval_results["bleu"],
            "sample_predictions": eval_results["sample_predictions"],
            "sample_references": eval_results["sample_references"]
        }, f, indent=2)

    return eval_results
