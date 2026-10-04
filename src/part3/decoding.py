"""
Part 3: Decoding Strategies from Scratch using model.forward()
Strictly without model.generate() from transformers.
Implements:
1. Greedy decoding
2. Top-k sampling (with temperature)
3. Top-p (Nucleus) sampling (with temperature)
4. Beam Search (parameterized for beam widths 1, 2, and 4)
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_logits(outputs: Any) -> torch.Tensor:
    """Robustly extracts 3D logits tensor (B, T, V) from model outputs."""
    if hasattr(outputs, "logits"):
        return outputs.logits
    elif isinstance(outputs, (tuple, list)):
        return outputs[0]
    elif isinstance(outputs, torch.Tensor):
        return outputs
    raise ValueError(f"Unsupported model output type: {type(outputs)}")


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 40,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None
) -> Tuple[torch.Tensor, float, List[float]]:
    """
    Greedy decoding: iteratively picks argmax of next-token logits.
    Returns:
        generated_ids: full sequence of input_ids + generated tokens
        elapsed_time: total execution time in seconds
        log_probs: log-probabilities of each generated token
    """
    start_time = time.perf_counter()
    curr_ids = input_ids.clone()
    token_log_probs: List[float] = []

    for _ in range(max_new_tokens):
        outputs = model(curr_ids)
        logits = _get_logits(outputs)
        next_token_logits = logits[:, -1, :]  # (B, V)

        log_probs = F.log_softmax(next_token_logits, dim=-1)
        max_log_prob, next_token = log_probs.max(dim=-1, keepdim=True)

        token_log_probs.append(max_log_prob.item())
        curr_ids = torch.cat([curr_ids, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    elapsed_time = time.perf_counter() - start_time
    return curr_ids, elapsed_time, token_log_probs


@torch.no_grad()
def top_k_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    k: int = 50,
    temperature: float = 1.0,
    max_new_tokens: int = 40,
    eos_token_id: Optional[int] = None
) -> Tuple[torch.Tensor, float, List[float]]:
    """
    Top-k sampling: restricts sampling pool to the top-k most likely tokens.
    """
    start_time = time.perf_counter()
    curr_ids = input_ids.clone()
    token_log_probs: List[float] = []

    for _ in range(max_new_tokens):
        outputs = model(curr_ids)
        logits = _get_logits(outputs)
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

        # Retain only top-k logits
        val_topk, _ = torch.topk(next_token_logits, min(k, next_token_logits.size(-1)))
        kth_threshold = val_topk[:, -1].unsqueeze(-1)
        filtered_logits = torch.where(
            next_token_logits < kth_threshold,
            torch.tensor(float("-inf"), device=next_token_logits.device),
            next_token_logits
        )

        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        log_probs = F.log_softmax(next_token_logits, dim=-1)
        selected_log_prob = log_probs.gather(1, next_token).item()
        token_log_probs.append(selected_log_prob)

        curr_ids = torch.cat([curr_ids, next_token], dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    elapsed_time = time.perf_counter() - start_time
    return curr_ids, elapsed_time, token_log_probs


@torch.no_grad()
def top_p_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
    max_new_tokens: int = 40,
    eos_token_id: Optional[int] = None
) -> Tuple[torch.Tensor, float, List[float]]:
    """
    Top-p (Nucleus) sampling: retains the smallest subset of tokens whose
    cumulative probability exceeds threshold p.
    """
    start_time = time.perf_counter()
    curr_ids = input_ids.clone()
    token_log_probs: List[float] = []

    for _ in range(max_new_tokens):
        outputs = model(curr_ids)
        logits = _get_logits(outputs)
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Mask tokens with cumulative probability above p
        sorted_indices_to_remove = cumulative_probs > p
        # Shift mask right to ensure at least the first token is kept
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter mask back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        filtered_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        log_probs = F.log_softmax(next_token_logits, dim=-1)
        selected_log_prob = log_probs.gather(1, next_token).item()
        token_log_probs.append(selected_log_prob)

        curr_ids = torch.cat([curr_ids, next_token], dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    elapsed_time = time.perf_counter() - start_time
    return curr_ids, elapsed_time, token_log_probs


class BeamHypothesis:
    """Stores individual beam candidate sequence and cumulative score."""
    def __init__(
        self,
        token_ids: torch.Tensor,
        score: float,
        token_scores: Optional[List[float]] = None,
        is_done: bool = False
    ):
        self.token_ids = token_ids  # 1D tensor (seq_len,)
        self.score = score          # Cumulative log probability
        self.token_scores = token_scores if token_scores is not None else []
        self.is_done = is_done

    def normalized_score(self, length_penalty: float = 0.7) -> float:
        """Computes length-normalized score to avoid short-sentence bias."""
        length = self.token_ids.size(0)
        penalty = ((5.0 + length) / 6.0) ** length_penalty
        return self.score / penalty


@torch.no_grad()
def beam_search_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    beam_width: int = 4,
    max_new_tokens: int = 40,
    length_penalty: float = 0.7,
    eos_token_id: Optional[int] = None
) -> Tuple[torch.Tensor, float, List[float]]:
    """
    Beam search decoding implemented from scratch with width 1, 2, or 4.
    Strictly uses model.forward().
    Returns best beam sequence, elapsed time, and per-token log probabilities.
    """
    start_time = time.perf_counter()
    device = input_ids.device
    initial_tokens = input_ids[0] if input_ids.ndim > 1 else input_ids  # (prompt_len,)

    hypotheses = [BeamHypothesis(token_ids=initial_tokens, score=0.0, token_scores=[], is_done=False)]
    completed: List[BeamHypothesis] = []

    for step in range(max_new_tokens):
        if len(hypotheses) == 0:
            break

        all_candidates: List[BeamHypothesis] = []

        for hyp in hypotheses:
            if hyp.is_done:
                completed.append(hyp)
                continue

            curr_input = hyp.token_ids.unsqueeze(0)  # (1, curr_len)
            outputs = model(curr_input)
            logits = _get_logits(outputs)
            next_logits = logits[0, -1, :]  # (V,)
            log_probs = F.log_softmax(next_logits, dim=-1)

            top_log_probs, top_indices = torch.topk(log_probs, k=beam_width)

            for i in range(beam_width):
                tok_id = top_indices[i].unsqueeze(0)
                tok_score = top_log_probs[i].item()
                new_tokens = torch.cat([hyp.token_ids, tok_id], dim=0)
                new_score = hyp.score + tok_score
                new_scores = hyp.token_scores + [tok_score]
                is_done = (eos_token_id is not None and tok_id.item() == eos_token_id)

                candidate = BeamHypothesis(
                    token_ids=new_tokens,
                    score=new_score,
                    token_scores=new_scores,
                    is_done=is_done
                )
                if is_done:
                    completed.append(candidate)
                else:
                    all_candidates.append(candidate)

        if len(all_candidates) == 0:
            break

        # Sort candidates by normalized score
        all_candidates.sort(key=lambda h: h.normalized_score(length_penalty), reverse=True)
        hypotheses = all_candidates[:beam_width]

        # Early termination: if completed hypotheses match beam_width and dominate active
        if len(completed) >= beam_width:
            best_completed_score = max(h.normalized_score(length_penalty) for h in completed)
            best_hyp_score = hypotheses[0].normalized_score(length_penalty)
            if best_completed_score >= best_hyp_score:
                break

    all_results = completed + hypotheses
    if not all_results:
        return input_ids.clone(), time.perf_counter() - start_time, [0.0]

    all_results.sort(key=lambda h: h.normalized_score(length_penalty), reverse=True)
    best_hyp = all_results[0]

    elapsed_time = time.perf_counter() - start_time
    best_tokens = best_hyp.token_ids.unsqueeze(0)
    scores = best_hyp.token_scores if len(best_hyp.token_scores) > 0 else [best_hyp.score]
    return best_tokens, elapsed_time, scores
