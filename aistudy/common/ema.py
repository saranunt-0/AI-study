"""Exponential moving average (EMA) of model weights.

EMA weights are a cheap, strong regulariser for generative models: almost every diffusion
model samples from EMA weights (decay 0.999-0.9999), and some LLM recipes average late
checkpoints for the same reason (see "checkpoint averaging" / model soups). Keep the EMA in
fp32 even when training in bf16.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self.shadow = copy.deepcopy(model).eval().float()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def _current_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        # Ramp the decay so early (bad) weights are forgotten quickly.
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = self._current_decay()
        model_params = dict(model.named_parameters())
        for name, p_ema in self.shadow.named_parameters():
            p_ema.lerp_(model_params[name].detach().float(), 1 - d)
        model_buffers = dict(model.named_buffers())
        for name, b_ema in self.shadow.named_buffers():
            b_ema.copy_(model_buffers[name])

    def state_dict(self) -> dict:
        return {"shadow": self.shadow.state_dict(), "num_updates": self.num_updates, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state["shadow"])
        self.num_updates = state.get("num_updates", 0)
        self.decay = state.get("decay", self.decay)
