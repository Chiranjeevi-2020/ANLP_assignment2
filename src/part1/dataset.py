"""
Dataset and Tokenization module for Part 1 (Vietnamese/Japanese to English Translation).
Handles belumind/en-vi-ja-curated-500k-triplets dataset.
Supports Byte-level BPE tokenization handling non-romanized Japanese and Vietnamese characters.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<vi>", "<ja>", "<en>"]


def build_or_load_tokenizer(
    vocab_size: int = 16000,
    save_path: str = "assets/tokenizer_multilingual.json",
    corpus_texts: Optional[List[str]] = None
) -> PreTrainedTokenizerFast:
    """
    Builds or loads a Byte-level BPE tokenizer capable of processing
    arbitrary UTF-8 bytes (Vietnamese diacritics and Japanese Kanji/Kana).
    """
    if os.path.exists(save_path):
        tokenizer = Tokenizer.from_file(save_path)
        fast_tok = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
        )
        return fast_tok

    # Build Byte-Level BPE tokenizer from scratch
    raw_tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw_tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False
    )

    if corpus_texts is not None and len(corpus_texts) > 0:
        raw_tokenizer.train_from_iterator(corpus_texts, trainer=trainer)
    else:
        # Fallback dummy seed training with typical tokens so tokenizer is valid
        seed_data = [
            "Hello world, this is English.",
            "こんにちは世界、日本語のテストです。",
            "Xin chào thế giới, đây là kiểm tra tiếng Việt.",
        ] * 100
        raw_tokenizer.train_from_iterator(seed_data, trainer=trainer)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    raw_tokenizer.save(save_path)

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    return fast_tok


class TranslationDataset(Dataset):
    """
    Dataset creating Vietnamese->English and Japanese->English samples.
    Each triplet (vi, ja, en) produces two training sequences:
    1. <vi> {vi_text} <en> {en_text} <eos>
    2. <ja> {ja_text} <en> {en_text} <eos>
    """
    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 256,
        loss_on_target_only: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.loss_on_target_only = loss_on_target_only
        self.data: List[Dict[str, Any]] = []

        for item in samples:
            en = item.get("en", "").strip()
            vi = item.get("vi", "").strip()
            ja = item.get("ja", "").strip()

            if vi and en:
                self.data.append({
                    "src_lang": "vi",
                    "src_text": vi,
                    "tgt_text": en,
                    "prompt": f"<vi> {vi} <en>",
                    "full_text": f"<vi> {vi} <en> {en} <eos>"
                })
            if ja and en:
                self.data.append({
                    "src_lang": "ja",
                    "src_text": ja,
                    "tgt_text": en,
                    "prompt": f"<ja> {ja} <en>",
                    "full_text": f"<ja> {ja} <en> {en} <eos>"
                })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        full_text = item["full_text"]
        prompt = item["prompt"]

        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = full_encoding["input_ids"].squeeze(0)  # (N_ctx,)

        targets = input_ids.clone()
        pad_token_id = self.tokenizer.pad_token_id or 0
        targets[targets == pad_token_id] = -100

        if self.loss_on_target_only:
            prompt_encoding = self.tokenizer(
                prompt,
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt"
            )
            prompt_len = min(prompt_encoding["input_ids"].shape[1], self.max_seq_len)
            targets[:prompt_len] = -100  # Mask out prompt tokens from loss computation

        return {
            "input_ids": input_ids,
            "targets": targets,
            "src_lang": item["src_lang"],
            "src_text": item["src_text"],
            "tgt_text": item["tgt_text"],
            "prompt": item["prompt"]
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([b["input_ids"] for b in batch])
    targets = torch.stack([b["targets"] for b in batch])
    src_langs = [b["src_lang"] for b in batch]
    src_texts = [b["src_text"] for b in batch]
    tgt_texts = [b["tgt_text"] for b in batch]
    prompts = [b["prompt"] for b in batch]

    return {
        "input_ids": input_ids,
        "targets": targets,
        "src_langs": src_langs,
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
        "prompts": prompts
    }
