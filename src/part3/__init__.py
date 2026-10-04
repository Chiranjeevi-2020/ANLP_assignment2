"""
Part 3: Decoding Strategies Module
"""

from .decoding import (
    greedy_decode,
    top_k_decode,
    top_p_decode,
    beam_search_decode,
    BeamHypothesis,
)
from .evaluate import (
    compute_generation_perplexity,
    compute_overlap_accuracy,
    run_decoding_benchmark,
    plot_beam_search_scaling,
)

__all__ = [
    "greedy_decode",
    "top_k_decode",
    "top_p_decode",
    "beam_search_decode",
    "BeamHypothesis",
    "compute_generation_perplexity",
    "compute_overlap_accuracy",
    "run_decoding_benchmark",
    "plot_beam_search_scaling",
]
