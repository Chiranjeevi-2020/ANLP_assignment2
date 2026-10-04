"""
Unit tests for Part 3: Decoding Strategies from Scratch.
Ensures greedy, top-k, top-p, and beam search decode correctly using only model.forward().
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.part3.decoding import (
    greedy_decode,
    top_k_decode,
    top_p_decode,
    beam_search_decode,
)


class DummyCausalLM(nn.Module):
    """
    Toy model that deterministically predicts next token based on simple linear mapping.
    Has no generate() method to guarantee decoding algorithms only use forward().
    """
    def __init__(self, vocab_size: int = 50, d_model: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)
        logits = self.head(x)
        return logits


def test_greedy_decode():
    torch.manual_seed(42)
    model = DummyCausalLM(vocab_size=20)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    gen_ids, elapsed, log_probs = greedy_decode(model, input_ids, max_new_tokens=5)
    assert gen_ids.shape == (1, 8), f"Expected shape (1, 8), got {gen_ids.shape}"
    assert len(log_probs) == 5
    assert elapsed > 0.0

    # Verify deterministic behavior: repeated calls yield identical sequence
    gen_ids_2, _, _ = greedy_decode(model, input_ids, max_new_tokens=5)
    assert torch.equal(gen_ids, gen_ids_2), "Greedy decoding must be deterministic"


def test_top_k_sampling():
    torch.manual_seed(42)
    model = DummyCausalLM(vocab_size=20)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    gen_ids, elapsed, _ = top_k_decode(model, input_ids, k=5, max_new_tokens=4)
    assert gen_ids.shape == (1, 7)
    assert elapsed > 0.0


def test_top_p_sampling():
    torch.manual_seed(42)
    model = DummyCausalLM(vocab_size=20)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    gen_ids, elapsed, _ = top_p_decode(model, input_ids, p=0.8, max_new_tokens=4)
    assert gen_ids.shape == (1, 7)
    assert elapsed > 0.0


@pytest.mark.parametrize("width", [1, 2, 4])
def test_beam_search_decode_widths(width):
    torch.manual_seed(42)
    model = DummyCausalLM(vocab_size=20)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    best_seq, elapsed, scores = beam_search_decode(
        model, input_ids, beam_width=width, max_new_tokens=4
    )
    assert best_seq.shape[0] == 1
    assert best_seq.shape[1] == 7
    assert elapsed > 0.0
    assert len(scores) > 0


def test_beam_1_matches_greedy():
    """Beam search with width 1 should yield the exact same sequence as greedy decoding."""
    torch.manual_seed(42)
    model = DummyCausalLM(vocab_size=20)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    greedy_ids, _, _ = greedy_decode(model, input_ids, max_new_tokens=6)
    beam_1_ids, _, _ = beam_search_decode(model, input_ids, beam_width=1, max_new_tokens=6, length_penalty=0.0)

    assert torch.equal(greedy_ids, beam_1_ids), f"Beam 1 {beam_1_ids} differs from greedy {greedy_ids}"


def test_beam_search_early_stopping_on_eos():
    """Verifies that beam search halts early when all hypotheses hit the EOS token."""
    torch.manual_seed(42)
    model = DummyCausalLM(vocab_size=20)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    out_1, _, _ = greedy_decode(model, input_ids, max_new_tokens=1)
    eos_candidate = out_1[0, -1].item()

    best_seq, elapsed, scores = beam_search_decode(
        model, input_ids, beam_width=2, max_new_tokens=20, eos_token_id=eos_candidate
    )
    assert best_seq.shape[1] < 23, f"Beam search failed to stop early on EOS: length={best_seq.shape[1]}"

