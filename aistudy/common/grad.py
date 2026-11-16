"""Gradient control and training-stability tooling.

What is here and when to use it
-------------------------------
global_grad_norm / clip_grad_norm_
    Log the global L2 norm EVERY step (pre-clip). It is the single most useful health signal.
    Clip at 1.0 for LLM pre-training (GPT-3, Llama, DeepSeek...). Clipping does not fix a
    bad setup; it caps the damage of rare outlier batches.
adaptive_grad_clip_
    Unit-wise Adaptive Gradient Clipping from NFNets (Brock et al., arXiv:2102.06171):
    clip each row's gradient relative to the row's weight norm. Useful for norm-free nets
    and large batch sizes.
SpikeGuard
    Skip the optimizer step when loss or grad-norm is a statistical outlier (or non-finite).
    Production runs combine this with "rewind to the last good checkpoint and skip the
    offending data" (PaLM, arXiv:2204.02311, section 5.1).
GradNoiseScale
    Online estimate of the "simple gradient noise scale" B_simple = tr(Sigma) / |G|^2
    (McCandlish et al., arXiv:1812.06162). B_simple approximates the critical batch size:
    below it, doubling the batch nearly halves the steps needed; above it, returns diminish.
per_parameter_grad_norms
    Find WHICH layer is blowing up (usually: embeddings, the last layers, attention logits).
z_loss
    PaLM's auxiliary loss  1e-4 * logsumexp(logits)^2 : keeps the softmax normaliser near 0,
    preventing logit drift that causes bf16 round-off and late-training divergence.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


def global_grad_norm(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    norms = torch._foreach_norm(grads) if hasattr(torch, "_foreach_norm") else [g.norm() for g in grads]
    return torch.linalg.vector_norm(torch.stack([n.float() for n in norms]))


def clip_grad_norm_(parameters: Iterable[torch.Tensor], max_norm: float) -> torch.Tensor:
    """Clip by global norm; returns the PRE-clip norm (what you want to log).

    max_norm <= 0 disables clipping but still returns the norm.
    """
    params = [p for p in parameters if p.grad is not None]
    if max_norm and max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    return global_grad_norm(params)


@torch.no_grad()
def adaptive_grad_clip_(parameters: Iterable[torch.Tensor], clip_factor: float = 0.01, eps: float = 1e-3) -> None:
    """Unit-wise AGC: g_i <- g_i * min(1, clip_factor * max(|w_i|, eps) / |g_i|).

    "Unit" = one output row of a weight matrix (dim 0), or the whole tensor for 1D params.
    """
    for p in parameters:
        if p.grad is None:
            continue
        if p.ndim > 1:
            dims = tuple(range(1, p.ndim))
            w_norm = p.norm(dim=dims, keepdim=True)
            g_norm = p.grad.norm(dim=dims, keepdim=True)
        else:
            w_norm, g_norm = p.norm(), p.grad.norm()
        max_norm = clip_factor * w_norm.clamp_min(eps)
        scale = (max_norm / g_norm.clamp_min(1e-6)).clamp(max=1.0)
        p.grad.mul_(scale)


class SpikeGuard:
    """Decide whether to skip an optimizer step because of a loss / grad-norm spike.

    Keeps an exponential moving mean and variance of log(grad_norm) and of the loss. A step
    is skipped when either z-score exceeds ``z_threshold`` (after ``warmup`` accepted steps)
    or when anything is NaN/Inf. Statistics are updated only with accepted steps. After
    ``max_consecutive`` skips in a row we stop skipping and warn: a persistent "spike" is a
    real distribution shift or a divergence - rewind to a checkpoint instead of hiding it.
    """

    def __init__(self, z_threshold: float = 6.0, beta: float = 0.98, warmup: int = 50, max_consecutive: int = 5):
        self.z, self.beta, self.warmup, self.max_consecutive = z_threshold, beta, warmup, max_consecutive
        self.stats = {"loss": [0.0, 1.0], "lgn": [0.0, 1.0]}  # [mean, var]
        self.accepted = 0
        self.consecutive = 0
        self.total_skipped = 0

    def _zscore(self, key: str, x: float) -> float:
        mean, var = self.stats[key]
        return (x - mean) / math.sqrt(var + 1e-12)

    def _update(self, key: str, x: float) -> None:
        mean, var = self.stats[key]
        if self.accepted == 0:
            self.stats[key] = [x, max(abs(x) * 0.1, 1e-4) ** 2]
            return
        b = self.beta
        delta = x - mean
        mean = mean + (1 - b) * delta
        var = b * (var + (1 - b) * delta * delta)
        self.stats[key] = [mean, var]

    def should_skip(self, loss: float, grad_norm: float) -> bool:
        if not (math.isfinite(loss) and math.isfinite(grad_norm)):
            self.consecutive += 1
            self.total_skipped += 1
            return True
        lgn = math.log(grad_norm + 1e-12)
        spike = False
        if self.accepted >= self.warmup:
            spike = self._zscore("loss", loss) > self.z or self._zscore("lgn", lgn) > self.z
        if spike and self.consecutive < self.max_consecutive:
            self.consecutive += 1
            self.total_skipped += 1
            return True
        if spike:
            print(f"[SpikeGuard] {self.consecutive} consecutive spikes; accepting step. "
                  "Consider rewinding to an earlier checkpoint and skipping this data.")
        self.consecutive = 0
        self._update("loss", loss)
        self._update("lgn", lgn)
        self.accepted += 1
        return False


class GradNoiseScale:
    """Online estimator of the simple gradient noise scale (McCandlish et al. 2018, App. A).

    Given squared gradient norms measured at two batch sizes b_small < b_big, unbiased
    estimates of the true |G|^2 and of tr(Sigma) are:
        |G|^2  ~ (b_big*|g_big|^2 - b_small*|g_small|^2) / (b_big - b_small)
        tr(S)  ~ (|g_small|^2 - |g_big|^2) / (1/b_small - 1/b_big)
    Each estimate is very noisy, so we EMA numerator and denominator separately.
    With gradient accumulation this is free: the first micro-batch is "small", the
    accumulated gradient is "big".
    """

    def __init__(self, beta: float = 0.95):
        self.beta = beta
        self.ema_g2: float | None = None
        self.ema_trace: float | None = None

    def update(self, small_sq: float, big_sq: float, b_small: float, b_big: float) -> float | None:
        if b_big <= b_small:
            return None
        g2 = (b_big * big_sq - b_small * small_sq) / (b_big - b_small)
        trace = (small_sq - big_sq) / (1.0 / b_small - 1.0 / b_big)
        if self.ema_g2 is None:
            self.ema_g2, self.ema_trace = g2, trace
        else:
            self.ema_g2 = self.beta * self.ema_g2 + (1 - self.beta) * g2
            self.ema_trace = self.beta * self.ema_trace + (1 - self.beta) * trace
        return self.value

    @property
    def value(self) -> float | None:
        if self.ema_g2 is None or self.ema_g2 <= 0:
            return None
        return self.ema_trace / self.ema_g2


def per_parameter_grad_norms(model: nn.Module, top_k: int = 5) -> list[tuple[str, float]]:
    norms = [(n, p.grad.detach().norm().item()) for n, p in model.named_parameters() if p.grad is not None]
    norms.sort(key=lambda x: x[1], reverse=True)
    return norms[:top_k]


def update_to_weight_ratio(model: nn.Module, prev: dict[str, torch.Tensor]) -> dict[str, float]:
    """|W_new - W_old| / |W_old| per parameter. Healthy LLM training sits around 1e-3 per step;
    much larger means the LR is too high for that tensor (Karpathy's classic diagnostic)."""
    out = {}
    for n, p in model.named_parameters():
        if n in prev:
            out[n] = ((p.detach() - prev[n]).norm() / (prev[n].norm() + 1e-12)).item()
    return out


def z_loss(logits: torch.Tensor) -> torch.Tensor:
    """mean(logsumexp(logits)^2) computed in fp32."""
    return torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
