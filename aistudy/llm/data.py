"""Tokenizer + data pipeline for LM pre-training.

We use a byte-level tokenizer (vocab = 256 bytes + BOS/EOS) so the examples need no
downloads and work on any language. Production models use BPE/Unigram tokenizers with
100k-260k entries (tiktoken, sentencepiece, HF tokenizers); swap ``ByteTokenizer`` for one of
those and nothing else changes.

Packing: documents are concatenated as  [BOS] doc [EOS] [BOS] doc [EOS] ...  into one long
token stream and the trainer samples random windows. This wastes zero compute on padding.
(Advanced: document-masked attention so tokens cannot attend across document boundaries.)

CLI (pre-tokenise large corpora once into uint16 .bin files that are memory-mapped):
    python -m aistudy.llm.data --input corpus/*.txt --out-dir data/mycorpus
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


class ByteTokenizer:
    BOS = 256
    EOS = 257
    vocab_size = 258

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [self.BOS] + ids
        if add_eos:
            ids = ids + [self.EOS]
        return ids

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids if int(i) < 256).decode("utf-8", errors="replace")


def default_corpus_files() -> list[str]:
    """With no --data flag we train on this repository's own markdown (self-contained demo)."""
    files = sorted(glob.glob(str(REPO_ROOT / "docs" / "*.md"))) + [str(REPO_ROOT / "README.md")]
    return [f for f in files if os.path.exists(f)]


def expand_paths(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        files.extend(m for m in matches if os.path.isfile(m))
    return files


def tokenize_files(files: list[str], tok: ByteTokenizer) -> np.ndarray:
    ids: list[int] = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            ids.extend(tok.encode(fh.read(), add_bos=True, add_eos=True))
    return np.asarray(ids, dtype=np.uint16)


class TokenStream:
    """Random-window sampler over a 1-D token array (in RAM or memory-mapped)."""

    def __init__(self, tokens: np.ndarray):
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def get_batch(self, batch_size: int, seq_len: int, device: torch.device,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.tokens) - seq_len - 1
        assert n > 0, f"corpus ({len(self.tokens)} tokens) shorter than seq_len {seq_len}"
        starts = torch.randint(0, n, (batch_size,), generator=generator).tolist()
        chunk = np.stack([self.tokens[s:s + seq_len + 1].astype(np.int64) for s in starts])
        chunk = torch.from_numpy(chunk)
        x, y = chunk[:, :-1], chunk[:, 1:]
        if device.type == "cuda":  # pinned memory + non_blocking overlaps H2D copy with compute
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def load_splits(data: list[str] | None, val_frac: float = 0.05) -> tuple[TokenStream, TokenStream]:
    """Return (train, val) streams.

    ``data`` may be: None (repo docs), a directory with train.bin/val.bin, or text files/globs.
    The validation split is the LAST ``val_frac`` of the stream (contiguous), so that random
    training windows never overlap validation text.
    """
    if data and len(data) == 1 and os.path.isdir(data[0]):
        d = data[0]
        train = np.memmap(os.path.join(d, "train.bin"), dtype=np.uint16, mode="r")
        val = np.memmap(os.path.join(d, "val.bin"), dtype=np.uint16, mode="r")
        return TokenStream(train), TokenStream(val)
    files = expand_paths(data) if data else default_corpus_files()
    if not files:
        raise FileNotFoundError("no training files found; pass --data path/to/text.txt")
    tokens = tokenize_files(files, ByteTokenizer())
    n_val = max(int(len(tokens) * val_frac), 1024)
    return TokenStream(tokens[:-n_val]), TokenStream(tokens[-n_val:])


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-tokenise text files into train.bin / val.bin")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-frac", type=float, default=0.01)
    args = p.parse_args()
    files = expand_paths(args.input)
    tokens = tokenize_files(files, ByteTokenizer())
    n_val = max(int(len(tokens) * args.val_frac), 1024)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens[:-n_val].tofile(os.path.join(args.out_dir, "train.bin"))
    tokens[-n_val:].tofile(os.path.join(args.out_dir, "val.bin"))
    print(f"{len(files)} files -> {len(tokens):,} tokens ({len(tokens) - n_val:,} train / {n_val:,} val)")


if __name__ == "__main__":
    main()
