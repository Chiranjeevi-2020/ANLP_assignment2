"""
Dataset module for Part 2: Pretraining on browndw/human-ai-parallel-corpus.
Implements next-token prediction streaming and chunking for Chinchilla-style training dynamics.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast


class PretrainDataset(Dataset):
    """
    Dataset for next-token prediction language modeling.
    Extracts fixed-length token chunks from continuous document texts.
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizerFast,
        seq_len: int = 256,
        prefix_eval_ratio: float = 0.5
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.prefix_eval_ratio = prefix_eval_ratio
        self.samples: List[torch.Tensor] = []
        self.eval_pairs: List[Tuple[str, str]] = []

        for text in texts:
            if not text or not text.strip():
                continue
            enc = tokenizer(text, truncation=False, return_tensors="pt")["input_ids"].squeeze(0)
            if len(enc) < 8:
                continue

            # Split into chunks of size seq_len
            for i in range(0, len(enc) - seq_len, seq_len):
                chunk = enc[i : i + seq_len]
                self.samples.append(chunk)

            # Store prefix and continuation for BLEU evaluation
            split_idx = int(len(text) * prefix_eval_ratio)
            prompt = text[:split_idx].strip()
            continuation = text[split_idx:].strip()
            if prompt and continuation:
                self.eval_pairs.append((prompt, continuation))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.samples[idx]
        return {
            "input_ids": chunk,
            "targets": chunk
        }
