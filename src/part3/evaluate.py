"""
Evaluation benchmark for Part 3 Decoding Strategies.
Loads EleutherAI/Pythia-160M and hamishivi/ROCStories.
Computes Perplexity, Accuracy, Latency, and saves Raw Generations.
"""

import os
import json
import time
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from .decoding import (
    greedy_decode,
    top_k_decode,
    top_p_decode,
    beam_search_decode,
    _get_logits,
)
from ..utils import compute_perplexity, compute_bleu

logger = logging.getLogger("Part3_Eval")


def compute_generation_perplexity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    gen_ids: torch.Tensor
) -> float:
    """
    Computes perplexity of the generated tokens conditioned on input_ids.
    PPL = exp(-1/T * sum log P(x_t | x_{<t}))
    """
    if gen_ids.size(1) <= input_ids.size(1):
        return 1.0

    prompt_len = input_ids.size(1)
    with torch.no_grad():
        outputs = model(gen_ids)
        logits = _get_logits(outputs)

        # Shift logits and labels for generated tokens
        shift_logits = logits[:, prompt_len - 1 : -1, :].contiguous()
        shift_labels = gen_ids[:, prompt_len:].contiguous()

        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return compute_perplexity(loss.item())


def compute_overlap_accuracy(pred_text: str, gold_text: str) -> float:
    """
    Computes token-level accuracy (Jaccard token overlap) against reference continuation.
    """
    pred_tokens = set(pred_text.lower().split())
    gold_tokens = set(gold_text.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    intersection = pred_tokens.intersection(gold_tokens)
    union = pred_tokens.union(gold_tokens)
    return float(len(intersection) / len(union) * 100.0)


def run_decoding_benchmark(
    model_name: str = "EleutherAI/Pythia-160M",
    dataset_name: str = "hamishivi/ROCStories",
    num_samples: int = 1000,
    device: Optional[torch.device] = None,
    output_dir: str = "assets/part3"
) -> Dict[str, Any]:
    """
    Evaluates greedy, top-k, top-p, and beam search (1, 2, 4) on ROCStories.
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    logger.info(f"Loading dataset: {dataset_name}")
    try:
        ds = load_dataset(dataset_name, split="test")
    except Exception as e:
        logger.warning(f"Could not load split 'test' directly ({e}), trying train/validation.")
        try:
            ds = load_dataset(dataset_name, split="train")
        except Exception as e2:
            logger.warning(f"Could not load Hugging Face dataset ({e2}). Using demonstration narrative corpus.")
            ds = [
                {"sentence1": "Tom wanted to learn to play the guitar.", "sentence2": "He bought a used guitar from a pawn shop.", "sentence3": "He practiced every single day for months.", "sentence4": "Soon his fingers were strong and agile.", "sentence5": "He played a song for his friends who applauded him."},
                {"sentence1": "Sarah went to the grocery store to buy ingredients.", "sentence2": "She needed flour, sugar, and cocoa powder.", "sentence3": "Back home, she mixed everything in a bowl.", "sentence4": "She baked the cake in the oven for thirty minutes.", "sentence5": "The chocolate cake turned out absolutely delicious."},
            ] * 500

    # Select num_samples with robust support for ROCStories schemas
    samples = []
    for item in ds:
        if len(samples) >= num_samples:
            break
        prompt, continuation = None, None
        if "sentence1" in item and "sentence5" in item:
            # Standard ROCStories five-sentence narrative structure
            s1 = (item.get("sentence1") or "").strip()
            s2 = (item.get("sentence2") or "").strip()
            s3 = (item.get("sentence3") or "").strip()
            s4 = (item.get("sentence4") or "").strip()
            s5 = (item.get("sentence5") or "").strip()
            prompt = f"{s1} {s2} {s3} {s4}".strip()
            continuation = s5
        elif "prompt" in item and "continuation" in item:
            prompt = (item.get("prompt") or "").strip()
            continuation = (item.get("continuation") or "").strip()
        elif "text" in item:
            text = (item.get("text") or "").strip()
            parts = [s.strip() for s in text.split(".") if s.strip()]
            if len(parts) >= 2:
                prompt = ". ".join(parts[:-1]) + "."
                continuation = parts[-1] + "."

        if prompt and continuation:
            samples.append((prompt, continuation))

    logger.info(f"Evaluating on {len(samples)} samples...")

    strategies = [
        {"name": "greedy", "fn": lambda m, ids: greedy_decode(m, ids, eos_token_id=tokenizer.eos_token_id)},
        {"name": "top_k", "fn": lambda m, ids: top_k_decode(m, ids, k=40, temperature=0.8, eos_token_id=tokenizer.eos_token_id)},
        {"name": "top_p", "fn": lambda m, ids: top_p_decode(m, ids, p=0.9, temperature=0.8, eos_token_id=tokenizer.eos_token_id)},
        {"name": "beam_1", "fn": lambda m, ids: beam_search_decode(m, ids, beam_width=1, eos_token_id=tokenizer.eos_token_id)},
        {"name": "beam_2", "fn": lambda m, ids: beam_search_decode(m, ids, beam_width=2, eos_token_id=tokenizer.eos_token_id)},
        {"name": "beam_4", "fn": lambda m, ids: beam_search_decode(m, ids, beam_width=4, eos_token_id=tokenizer.eos_token_id)},
    ]

    results_summary: Dict[str, Dict[str, Any]] = {}
    raw_generations: Dict[str, List[Dict[str, Any]]] = {}

    for strat in strategies:
        strat_name = strat["name"]
        logger.info(f"--- Running decoding strategy: {strat_name} ---")

        total_time = 0.0
        perplexities = []
        accuracies = []
        bleu_preds = []
        bleu_refs = []
        strat_generations = []

        for prompt, gold_continuation in tqdm(samples, desc=strat_name, leave=False):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]

            gen_ids, elapsed, _ = strat["fn"](model, input_ids)
            total_time += elapsed

            pred_tokens = gen_ids[0, input_ids.shape[1]:]
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

            ppl = compute_generation_perplexity(model, input_ids, gen_ids)
            acc = compute_overlap_accuracy(pred_text, gold_continuation)

            perplexities.append(ppl)
            accuracies.append(acc)
            bleu_preds.append(pred_text)
            bleu_refs.append([gold_continuation])

            strat_generations.append({
                "prompt": prompt,
                "gold_continuation": gold_continuation,
                "generated": pred_text,
                "ppl": ppl,
                "acc": acc,
                "time_sec": elapsed
            })

        avg_ppl = float(np.mean([p for p in perplexities if not math.isinf(p)])) if perplexities else float("nan")
        avg_acc = float(np.mean(accuracies))
        bleu_score = compute_bleu(bleu_preds, bleu_refs)

        results_summary[strat_name] = {
            "mean_perplexity": avg_ppl,
            "mean_accuracy": avg_acc,
            "bleu_score": bleu_score,
            "total_time_seconds": total_time,
            "seconds_per_sample": total_time / len(samples)
        }
        raw_generations[strat_name] = strat_generations

    # Save outputs
    with open(os.path.join(output_dir, "decoding_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)

    with open(os.path.join(output_dir, "raw_generations.json"), "w") as f:
        json.dump(raw_generations, f, indent=2)

    # Plot beam search runtime scaling
    plot_beam_search_scaling(results_summary, save_path=os.path.join(output_dir, "beam_search_runtime.png"))

    return {"summary": results_summary, "raw_generations": raw_generations}


def plot_beam_search_scaling(
    results: Dict[str, Dict[str, Any]],
    save_path: str = "assets/beam_search_runtime.png"
) -> None:
    """Plots beam search latency vs beam width (1, 2, 4)."""
    beam_widths = [1, 2, 4]
    times = []
    for w in beam_widths:
        key = f"beam_{w}"
        if key in results:
            times.append(results[key]["total_time_seconds"])

    if len(times) == len(beam_widths):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.figure(figsize=(7, 5))
        plt.plot(beam_widths, times, marker="o", color="crimson", linewidth=2.5, markersize=8)
        plt.title("Beam Search Execution Time vs Beam Width")
        plt.xlabel("Beam Width (B)")
        plt.ylabel("Execution Time (Seconds)")
        plt.xticks(beam_widths)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        logger.info(f"Beam search scaling plot saved to {save_path}")
