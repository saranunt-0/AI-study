"""Decoding algorithms: temperature / top-k / top-p / min-p sampling, KV-cached generation,
and speculative decoding.

Speculative decoding (Leviathan et al. arXiv:2211.17192; Chen et al. arXiv:2302.01318)
------------------------------------------------------------------------------------------
A cheap draft model proposes ``gamma`` tokens; the expensive target model scores all of them
in ONE forward pass; each draft token d_i is accepted with probability min(1, p(d_i)/q(d_i)),
and on the first rejection we resample from norm(max(0, p - q)). This provably samples from
exactly the target distribution p - the speed-up is free of quality loss. Modern variants
draft with heads attached to the target itself (Medusa, EAGLE-3, DeepSeek-V3's MTP module).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def logits_to_probs(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                    top_p: float = 1.0, min_p: float = 0.0) -> torch.Tensor:
    """Turn logits [..., V] into a sampling distribution.

    * temperature == 0  -> greedy (one-hot argmax), handled uniformly as a distribution.
    * top_k             -> keep the k most likely tokens.
    * top_p (nucleus)   -> keep the smallest set with cumulative prob >= top_p.
    * min_p             -> keep tokens with p >= min_p * p_max (Nguyen et al. arXiv:2407.01082);
                           adapts to the model's confidence, robust at high temperature.
    """
    logits = logits.float()
    if temperature <= 0:
        return F.one_hot(logits.argmax(-1), logits.size(-1)).float()
    logits = logits / temperature
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if min_p > 0:
        probs = probs.masked_fill(probs < min_p * probs.amax(-1, keepdim=True), 0.0)
    if top_p < 1.0:
        sorted_p, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = sorted_p.cumsum(-1)
        remove = (cum - sorted_p) > top_p  # keep the token that crosses the threshold
        sorted_p = sorted_p.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_p)
    return probs / probs.sum(-1, keepdim=True)


@torch.no_grad()
def generate(model, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0,
             top_p: float = 1.0, min_p: float = 0.0, eos_id: int | None = None, use_cache: bool = True,
             generator: torch.Generator | None = None,
             on_token: Callable[[torch.Tensor], None] | None = None) -> torch.Tensor:
    """Autoregressive sampling. prompt: [B, T0] LongTensor. Returns [B, T0 + n_generated]."""
    model.eval()
    B, T0 = prompt.shape
    max_len = model.cfg.max_seq_len
    seq = prompt
    caches = None
    if use_cache:
        assert T0 < max_len, "prompt must be shorter than max_seq_len when using the KV cache"
        caches = model.make_kv_cache(B, max_len)
        logits = model(prompt, caches=caches, start_pos=0, last_only=True)["logits"][:, -1]  # prefill
    else:
        logits = model(prompt[:, -max_len:], last_only=True)["logits"][:, -1]
    finished = torch.zeros(B, dtype=torch.bool, device=prompt.device)
    for i in range(max_new_tokens):
        probs = logits_to_probs(logits, temperature, top_k, top_p, min_p)
        nxt = torch.multinomial(probs, 1, generator=generator)
        if eos_id is not None:
            nxt = torch.where(finished[:, None], torch.full_like(nxt, eos_id), nxt)
            finished |= nxt[:, 0] == eos_id
        seq = torch.cat([seq, nxt], dim=1)
        if on_token is not None:
            on_token(nxt)
        if (eos_id is not None and bool(finished.all())) or i == max_new_tokens - 1:
            break
        if use_cache:
            if seq.size(1) >= max_len:
                break
            logits = model(nxt, caches=caches, start_pos=seq.size(1) - 1, last_only=True)["logits"][:, -1]
        else:  # recompute everything each step (O(T^2) work) - only for comparison
            logits = model(seq[:, -max_len:], last_only=True)["logits"][:, -1]
    return seq


@torch.no_grad()
def speculative_generate(target, draft, prompt: torch.Tensor, max_new_tokens: int, gamma: int = 4,
                         temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0,
                         generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """Speculative sampling for batch size 1. Returns (sequence [1, T], stats)."""
    assert prompt.size(0) == 1, "speculative_generate supports batch size 1"
    target.eval(), draft.eval()
    device = prompt.device
    max_len = min(target.cfg.max_seq_len, draft.cfg.max_seq_len)
    t_cache, d_cache = target.make_kv_cache(1, max_len), draft.make_kv_cache(1, max_len)
    seq = prompt[0].tolist()
    T0 = len(seq)
    if T0 > 1:  # prefill everything except the last token; caches hold seq[:-1]
        target(prompt[:, :-1], caches=t_cache, start_pos=0)
        draft(prompt[:, :-1], caches=d_cache, start_pos=0)
    t_valid = d_valid = T0 - 1        # number of correct entries in each cache
    drafted = accepted = rounds = 0

    def probs(logits):
        return logits_to_probs(logits, temperature, top_k, top_p, min_p)

    while len(seq) - T0 < max_new_tokens and len(seq) + gamma + 1 <= max_len:
        rounds += 1
        # 1) draft gamma tokens autoregressively with the small model
        q_dists, drafts = [], []
        feed = seq[d_valid:]
        for _ in range(gamma):
            out = draft(torch.tensor([feed], device=device), caches=d_cache, start_pos=d_valid, last_only=True)
            d_valid += len(feed)
            q = probs(out["logits"][0, -1])
            tok = int(torch.multinomial(q, 1, generator=generator))
            q_dists.append(q)
            drafts.append(tok)
            feed = [tok]
        # 2) score [last token, d_1..d_gamma] with the target in a single forward pass
        feed_t = seq[t_valid:] + drafts
        out = target(torch.tensor([feed_t], device=device), caches=t_cache, start_pos=t_valid)
        p_dists = probs(out["logits"][0, -(gamma + 1):])  # p_dists[j] predicts the token after d_j
        # 3) accept / reject
        new_tokens = []
        for j, tok in enumerate(drafts):
            ratio = (p_dists[j, tok] / q_dists[j][tok]).item()
            if torch.rand(1, generator=generator).item() < min(1.0, ratio):
                new_tokens.append(tok)
                continue
            residual = (p_dists[j] - q_dists[j]).clamp_min(0.0)
            residual = residual / residual.sum() if residual.sum() > 0 else p_dists[j]
            new_tokens.append(int(torch.multinomial(residual, 1, generator=generator)))
            break
        else:  # every draft accepted -> one bonus token from the target for free
            new_tokens.append(int(torch.multinomial(p_dists[gamma], 1, generator=generator)))
        n_acc = len(new_tokens) - 1
        drafted += gamma
        accepted += n_acc
        seq += new_tokens
        t_valid = len(seq) - 1                  # target cache = old seq + accepted drafts
        d_valid = min(d_valid, len(seq) - 1)    # draft cache may lag one token behind
    seq = seq[:T0 + max_new_tokens]
    stats = {"rounds": rounds, "acceptance_rate": accepted / max(1, drafted),
             "tokens_per_target_call": (len(seq) - T0) / max(1, rounds)}
    return torch.tensor([seq], device=device), stats
