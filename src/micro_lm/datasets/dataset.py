"""
Dataset and tokenizer for data science Q&A. Uses a simple word-level tokenizer
so we only depend on PyTorch and the standard library (open source).
"""
import csv
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

# Special tokens (must match config)
PAD = "<pad>"
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"


class SimpleTokenizer:
    """Word-level tokenizer with special tokens. Builds vocab from a list of texts."""

    def __init__(self, max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.token2id = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
        self.id2token = {0: PAD, 1: UNK, 2: SOS, 3: EOS}

    def _tokenize(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text.strip().lower())
        return text.split()

    def build_vocab(self, texts: list[str]) -> None:
        from collections import Counter

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self._tokenize(text))
        for tok, _ in counter.most_common(self.max_vocab_size - len(self.token2id)):
            if tok not in self.token2id:
                self.token2id[tok] = len(self.token2id)
                self.id2token[self.token2id[tok]] = tok

    def encode(self, text: str, add_sos_eos: bool = False) -> list[int]:
        toks = self._tokenize(text)
        ids = [self.token2id.get(t, self.token2id[UNK]) for t in toks]
        if add_sos_eos:
            ids = [self.token2id[SOS]] + ids + [self.token2id[EOS]]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special = {self.token2id[PAD], self.token2id[UNK], self.token2id[SOS], self.token2id[EOS]}
        toks = [
            self.id2token.get(i, UNK)
            for i in ids
            if not (skip_special and i in special)
        ]
        return " ".join(toks)

    @property
    def pad_id(self) -> int:
        return self.token2id[PAD]

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)


def collate_qa(
    batch: list[dict],
    tokenizer: SimpleTokenizer,
    max_src_len: int,
    max_tgt_len: int,
) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_id
    src_ids = [b["src_ids"] for b in batch]
    tgt_ids = [b["tgt_ids"] for b in batch]

    def pad_and_stack(ids_list: list[list[int]], max_len: int) -> torch.Tensor:
        padded = []
        for ids in ids_list:
            if len(ids) > max_len:
                ids = ids[:max_len]
            padded.append(ids + [pad_id] * (max_len - len(ids)))
        return torch.tensor(padded, dtype=torch.long)

    src = pad_and_stack(src_ids, max_src_len)
    tgt = pad_and_stack(tgt_ids, max_tgt_len)
    return {"src": src, "tgt": tgt}


class ScienceQADataset(Dataset):
    """Dataset of (question, answer) pairs from a CSV with columns 'question' and 'answer'."""

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: SimpleTokenizer,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
        build_vocab: bool = True,
    ):
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer
        self.pairs: list[tuple[str, str]] = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "question" not in reader.fieldnames or "answer" not in reader.fieldnames:
                raise ValueError("CSV must have 'question' and 'answer' columns")
            for row in reader:
                q = (row.get("question") or "").strip()
                a = (row.get("answer") or "").strip()
                if q and a:
                    self.pairs.append((q, a))

        if build_vocab:
            all_texts = [q for q, _ in self.pairs] + [a for _, a in self.pairs]
            tokenizer.build_vocab(all_texts)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict[str, Any]:
        q, a = self.pairs[i]
        src_ids = self.tokenizer.encode(q)[: self.max_src_len]
        # Target: SOS + answer + EOS (decoder input is shifted inside train loop)
        tgt_ids = self.tokenizer.encode(a, add_sos_eos=True)[: self.max_tgt_len]
        return {"src_ids": src_ids, "tgt_ids": tgt_ids, "question": q, "answer": a}


def build_tokenizer_and_dataset(
    csv_path: str | Path,
    max_src_len: int = 128,
    max_tgt_len: int = 128,
    max_vocab_size: int = 10000,
) -> tuple[SimpleTokenizer, ScienceQADataset]:
    tokenizer = SimpleTokenizer(max_vocab_size=max_vocab_size)
    dataset = ScienceQADataset(
        csv_path,
        tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        build_vocab=True,
    )
    return tokenizer, dataset
