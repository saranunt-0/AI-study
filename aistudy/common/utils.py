"""Small, dependency-free utilities shared by every training script."""

from __future__ import annotations

import contextlib
import json
import os
import random
import struct
import time
import zlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


# --------------------------------------------------------------------------------------
# Reproducibility & devices
# --------------------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (all devices).

    ``deterministic=True`` additionally forces deterministic kernels. It is slower and some
    ops will raise; use it to debug "is this a bug or is it noise?" questions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, dtype: str = "bf16"):
    """Mixed-precision context.

    * bf16: same exponent range as fp32, so no loss scaling is needed. Default on Ampere+.
    * fp16: needs ``torch.amp.GradScaler`` to avoid gradient underflow.
    * fp32: reference / debugging.

    Master weights and optimizer state stay in fp32 either way; only matmuls/activations
    are run in low precision. On CPU we default to fp32 because bf16 matmuls are slow there.
    """
    if dtype == "fp32":
        return contextlib.nullcontext()
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    return torch.autocast(device_type=device.type, dtype=torch_dtype)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# --------------------------------------------------------------------------------------
# Distributed (torchrun) helpers
# --------------------------------------------------------------------------------------
def init_distributed() -> tuple[bool, int, int, int]:
    """Initialise torch.distributed if launched with torchrun.

    Returns ``(is_distributed, rank, local_rank, world_size)``. Uses NCCL on GPU and gloo on
    CPU, so ``torchrun --nproc_per_node=2 -m aistudy.llm.train`` also works on a laptop.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def save_checkpoint(path: str | os.PathLike, **items: Any) -> None:
    """Atomically save a checkpoint (write to tmp file, then rename).

    Atomic writes matter at scale: a job killed mid-write must never corrupt the only
    checkpoint you have. Pass modules/optimizers directly; their ``state_dict`` is taken.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, value in items.items():
        if hasattr(value, "state_dict"):
            payload[key] = value.state_dict()
        else:
            payload[key] = _to_jsonable(value)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | os.PathLike, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
class JsonlLogger:
    """Print a compact line and append the full record to a .jsonl file.

    JSONL is trivially greppable and loads into pandas with ``pd.read_json(path, lines=True)``.
    Swap in wandb / tensorboard in production; keep a local copy regardless.
    """

    def __init__(self, path: str | os.PathLike | None, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(path) if path else None
        if self.enabled and self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def log(self, record: dict, print_keys: tuple[str, ...] | None = None) -> None:
        if not self.enabled:
            return
        record = {"time": round(time.time() - self._t0, 2), **record}
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
        keys = print_keys or tuple(record.keys())
        parts = []
        for k in keys:
            if k not in record:
                continue
            v = record[k]
            parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
        print(" | ".join(parts), flush=True)


# --------------------------------------------------------------------------------------
# Images (tiny PNG writer so we do not depend on PIL / matplotlib)
# --------------------------------------------------------------------------------------
def _write_png(path: str | os.PathLike, img: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 array as PNG using only zlib."""
    h, w, c = img.shape
    assert c == 3 and img.dtype == np.uint8
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)


def save_image_grid(images: torch.Tensor, path: str | os.PathLike, nrow: int = 8,
                    value_range: tuple[float, float] = (-1.0, 1.0), upscale: int = 1, pad: int = 1) -> None:
    """Save a batch of images (N, C, H, W) as a single PNG grid. C may be 1 or 3."""
    images = images.detach().float().cpu()
    lo, hi = value_range
    images = ((images - lo) / (hi - lo)).clamp(0, 1)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if upscale > 1:
        images = images.repeat_interleave(upscale, dim=2).repeat_interleave(upscale, dim=3)
    n, c, h, w = images.shape
    ncol = min(nrow, n)
    nrows = (n + ncol - 1) // ncol
    grid = torch.ones(c, nrows * (h + pad) + pad, ncol * (w + pad) + pad)
    for i in range(n):
        r, col = divmod(i, ncol)
        y0, x0 = pad + r * (h + pad), pad + col * (w + pad)
        grid[:, y0:y0 + h, x0:x0 + w] = images[i]
    arr = (grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    _write_png(path, arr)
