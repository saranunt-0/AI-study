"""Optimizers and learning-rate schedules.

Contents
--------
* ``build_optimizer`` - AdamW with correct weight-decay groups, or Muon (+AdamW for the rest).
* ``Muon``            - MomentUm Orthogonalized by Newton-Schulz (Keller Jordan, 2024), with
                        the "match AdamW RMS" update scaling and decoupled weight decay from
                        Moonshot's "Muon is Scalable for LLM Training" (arXiv:2502.16982).
* ``lr_schedule``     - linear warmup followed by cosine, WSD (warmup-stable-decay) or constant.

Why weight-decay groups matter: decaying RMSNorm gains or biases pulls them toward 0 and
silently hurts quality. The convention used by nanoGPT, Llama, OLMo...: decay every tensor
with ndim >= 2 (matrices, embeddings) and nothing else.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
from torch import nn


# --------------------------------------------------------------------------------------
# Muon
# --------------------------------------------------------------------------------------
@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalise G (i.e. map G = U S V^T  ->  U V^T).

    Quintic Newton-Schulz iteration with coefficients tuned to maximise the slope at zero
    (Keller Jordan). It does not converge to exactly U V^T; singular values land roughly in
    [0.7, 1.2], which empirically does not hurt. Runs in bf16 on GPU.
    """
    assert G.ndim == 2, "Muon expects 2D parameters (flatten conv kernels first)"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16) if G.is_cuda else G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)  # spectral norm <= Frobenius norm <= 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2D hidden-layer weights + AdamW for everything else, in one optimizer.

    Each param group must contain ``use_muon: bool``.

    Muon groups: momentum -> Nesterov -> Newton-Schulz orthogonalisation -> scale -> step.
    The update is an (approximately) semi-orthogonal matrix: every singular direction moves
    by the same amount, which prevents a few dominant directions from eating the update
    (the failure mode of SGD/Adam on matrices). ``scale_mode``:
      * ``"moonlight"`` (default): update *= 0.2*sqrt(max(fan_out, fan_in)), giving an update
        RMS of ~0.2 like AdamW, so you can reuse AdamW learning rates and weight decay.
      * ``"keller"``: update *= sqrt(max(1, fan_out/fan_in)); use lr ~ 0.02.

    Do NOT apply Muon to embeddings, the LM head, norms or biases - they are not "matrices
    acting on hidden states" and are better served by AdamW (see ``build_optimizer``).
    """

    def __init__(self, param_groups, lr: float = 1e-3, weight_decay: float = 0.1,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
                 scale_mode: str = "moonlight"):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, betas=betas, eps=eps, scale_mode=scale_mode, use_muon=False)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.lerp_(p.grad, 1 - group["momentum"])
                    g = p.grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                    g2d = g.reshape(g.size(0), -1)
                    update = zeropower_via_newtonschulz5(g2d, steps=group["ns_steps"])
                    fan_out, fan_in = g2d.shape
                    if group["scale_mode"] == "moonlight":
                        update = update * (0.2 * math.sqrt(max(fan_out, fan_in)))
                    else:
                        update = update * math.sqrt(max(1.0, fan_out / fan_in))
                    p.mul_(1 - lr * wd)  # decoupled weight decay
                    p.add_(update.reshape(p.shape), alpha=-lr)
            else:  # plain AdamW
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    state["exp_avg"].lerp_(p.grad, 1 - beta1)
                    state["exp_avg_sq"].lerp_(p.grad.square(), 1 - beta2)
                    denom = (state["exp_avg_sq"] / (1 - beta2 ** t)).sqrt_().add_(group["eps"])
                    p.mul_(1 - lr * wd)
                    p.addcdiv_(state["exp_avg"], denom, value=-lr / (1 - beta1 ** t))
        return loss


# --------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------
def _default_muon_exclude(name: str, p: nn.Parameter) -> bool:
    """Parameters that should stay on AdamW even though they are 2D."""
    keys = ("embed", "lm_head", "pos_emb", "router", "head_out", "final_layer", "x_embed",
            "t_embed", "y_embed", "patch")
    return any(k in name for k in keys)


def build_optimizer(model: nn.Module, kind: str = "adamw", lr: float = 3e-4, weight_decay: float = 0.1,
                    betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
                    muon_exclude: Callable[[str, nn.Parameter], bool] = _default_muon_exclude,
                    fused: bool | None = None) -> torch.optim.Optimizer:
    """Create an optimizer with sensible parameter groups.

    betas=(0.9, 0.95) rather than PyTorch's default 0.999 is standard for LLM pre-training:
    a shorter second-moment memory reacts faster to gradient-scale changes and is less prone
    to the "stale denominator" spikes described in Wortsman et al. 2023 (arXiv:2309.14322).
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if kind == "adamw":
        decay = [p for n, p in named if p.ndim >= 2]
        no_decay = [p for n, p in named if p.ndim < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if fused is None:
            fused = torch.cuda.is_available() and all(p.is_cuda for _, p in named)
        opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)
    elif kind == "muon":
        muon_params = [p for n, p in named if p.ndim >= 2 and not muon_exclude(n, p)]
        adam_decay = [p for n, p in named if p.ndim >= 2 and muon_exclude(n, p)]
        adam_no_decay = [p for n, p in named if p.ndim < 2]
        groups = [
            {"params": muon_params, "use_muon": True, "weight_decay": weight_decay},
            {"params": adam_decay, "use_muon": False, "weight_decay": weight_decay},
            {"params": adam_no_decay, "use_muon": False, "weight_decay": 0.0},
        ]
        groups = [g for g in groups if g["params"]]
        opt = Muon(groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    else:
        raise ValueError(f"unknown optimizer {kind!r}")
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt


# --------------------------------------------------------------------------------------
# Learning-rate schedules
# --------------------------------------------------------------------------------------
def lr_schedule(step: int, total_steps: int, warmup_steps: int, kind: str = "cosine",
                min_ratio: float = 0.1, decay_frac: float = 0.2) -> float:
    """Return the LR multiplier in [0, 1] for ``step`` (0-indexed).

    * cosine   : warmup -> half-cosine down to ``min_ratio``. Needs total_steps up front.
    * wsd      : warmup -> constant -> linear decay over the last ``decay_frac`` of training
                 (MiniCPM arXiv:2404.06395; Hagele et al. arXiv:2405.18392). You can branch
                 a decay from any stable-phase checkpoint, which makes data/length ablations
                 and "continue training later" cheap. Used by DeepSeek, Kimi, SmolLM3...
    * constant : warmup -> constant (useful for fine-tuning and debugging).

    Warmup exists because Adam's second-moment estimate is noisy for the first steps and the
    early loss landscape is sharp; a large LR at step 0 is the #1 cause of early divergence.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if kind == "constant":
        return 1.0
    if kind == "cosine":
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    if kind == "wsd":
        decay_steps = max(1, int(decay_frac * total_steps))
        decay_start = total_steps - decay_steps
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / decay_steps)
        return 1.0 - (1.0 - min_ratio) * progress
    raise ValueError(f"unknown schedule {kind!r}")


def set_lr(optimizer: torch.optim.Optimizer, multiplier: float) -> float:
    """Apply ``initial_lr * multiplier`` to every group; returns the first group's LR."""
    for g in optimizer.param_groups:
        g["lr"] = g["initial_lr"] * multiplier
    return optimizer.param_groups[0]["lr"]


def parameters_with_grad(params: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    return [p for p in params if p.grad is not None]
