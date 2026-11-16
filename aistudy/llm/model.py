"""A modern decoder-only Transformer language model, written to be read.

Architecture (what a 2025-2026 open-weights LLM block looks like)
------------------------------------------------------------------
    x ─┬─ RMSNorm ─ Attention(GQA, RoPE, QK-norm, output gate) ─(+)─┬─ RMSNorm ─ FFN ─(+)─▶
       └──────────────────────────────────────────────────────────┘ └──────────────────┘
    FFN = SwiGLU (dense)  or  Mixture-of-Experts (fine-grained routed experts + shared expert)

Component -> where it comes from -> why
* Pre-norm + RMSNorm (Zhang & Sennrich 2019)            stable deep residual stacks, cheaper than LayerNorm
* RoPE (Su et al. 2021, arXiv:2104.09864)               relative positions, extrapolation tricks (YaRN)
* GQA (Ainslie et al. 2023, arXiv:2305.13245)           n_kv_heads < n_heads -> smaller KV cache
* QK-norm (OLMo 2, Qwen3, Gemma 3)                      bounds attention logits -> fewer loss spikes
* Gated attention (Qwen, NeurIPS'25 best paper)         sigmoid gate on SDPA output: removes attention
                                                         sinks / massive activations, fewer spikes
* SwiGLU (Shazeer 2020, arXiv:2002.05202)               better quality per FLOP than GELU MLP
* MoE (DeepSeekMoE arXiv:2401.06066, DeepSeek-V3)       sigmoid router, fine-grained + shared experts,
                                                         auxiliary-loss-free balancing via routing bias
* z-loss (PaLM) / logit soft-cap (Gemma 2)              keep output logits bounded
* Weight tying, scaled init of residual projections     GPT-2 style; std / sqrt(2 * n_layers)

What frontier models add on top (see docs/01-language-models.md): MLA / sparse attention
(DeepSeek), hybrid linear-attention layers (Qwen3-Next/3.5 Gated DeltaNet, Kimi Linear),
multi-token prediction, hyper-connections (mHC), FP8/FP4 kernels, expert parallelism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from aistudy.common.grad import z_loss as _z_loss

_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
_SDPA_HAS_GQA = _TORCH_VERSION >= (2, 5)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class LLMConfig:
    vocab_size: int = 264            # 256 bytes + BOS/EOS, padded to a multiple of 8
    max_seq_len: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2              # GQA; == n_heads -> MHA, == 1 -> MQA
    d_ff: int | None = None          # SwiGLU hidden size; default ~8/3 * d_model
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    qk_norm: bool = True
    attn_gate: bool = True
    tie_embeddings: bool = True
    dropout: float = 0.0
    logit_softcap: float = 0.0       # e.g. 30.0 (Gemma 2); 0 disables
    z_loss_coef: float = 1e-4
    init_std: float = 0.02
    # Mixture of Experts (n_experts = 0 -> dense model)
    n_experts: int = 0
    n_shared_experts: int = 1
    moe_top_k: int = 2
    moe_d_ff: int | None = None      # per-expert hidden size; default d_ff // moe_top_k
    moe_first_dense: int = 1         # first layers stay dense (DeepSeek-V3 keeps 3)
    moe_aux_loss_coef: float = 0.0   # classic load-balance loss; 0 -> rely on bias balancing
    moe_bias_update_rate: float = 1e-3
    grad_checkpointing: bool = False

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert (self.d_model // self.n_heads) % 2 == 0, "head_dim must be even for RoPE"
        if self.d_ff is None:
            self.d_ff = 64 * math.ceil(8 / 3 * self.d_model / 64)
        if self.n_experts and self.moe_d_ff is None:
            self.moe_d_ff = max(32, self.d_ff // self.moe_top_k)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """y = x / rms(x) * g. Computed in fp32 for stability, cast back to the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).type_as(x)


def precompute_rope(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)  # [T, D/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x_i, x_{i+D/2}) by position-dependent angles. x: [B, T, H, D]."""
    d = x.shape[-1] // 2
    xf = x.float()
    x1, x2 = xf[..., :d], xf[..., d:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.type_as(x)


class KVCache:
    """Pre-allocated key/value cache for one layer: [B, n_kv_heads, max_len, head_dim].

    Positions >= the current length are garbage; "rolling back" (needed for speculative
    decoding) is therefore free - just restart writing from an earlier position.
    Production engines (vLLM) page this memory in fixed-size blocks to avoid fragmentation.
    """

    def __init__(self, batch_size: int, n_kv_heads: int, max_len: int, head_dim: int,
                 device: torch.device, dtype: torch.dtype):
        shape = (batch_size, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len

    def update(self, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = start_pos + k.size(2)
        assert end <= self.max_len, "KV cache overflow"
        self.k[:, :, start_pos:end] = k
        self.v[:, :, start_pos:end] = v
        return self.k[:, :, :end], self.v[:, :, :end]


def _attention_mask(q_len: int, kv_len: int, start_pos: int, device) -> tuple[torch.Tensor | None, bool]:
    """Return (mask, is_causal) for SDPA. Query i sits at absolute position start_pos + i."""
    if q_len == kv_len:
        return None, True                      # ordinary causal self-attention
    if q_len == 1:
        return None, False                     # one new token attends to the whole cache
    q_pos = torch.arange(start_pos, start_pos + q_len, device=device)[:, None]
    k_pos = torch.arange(kv_len, device=device)[None, :]
    return k_pos <= q_pos, False               # chunked prefill / speculative verification


class Attention(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        H, Hkv, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, H * D, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, Hkv * D, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, cfg.d_model, bias=False)
        self.gate_proj = nn.Linear(cfg.d_model, H * D, bias=False) if cfg.attn_gate else None
        self.q_norm = RMSNorm(D, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(D, cfg.norm_eps) if cfg.qk_norm else None

    def forward(self, x, cos, sin, cache: KVCache | None = None, start_pos: int = 0):
        B, T, _ = x.shape
        H, Hkv, D = self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim
        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, Hkv, D)
        v = self.v_proj(x).view(B, T, Hkv, D)
        if self.q_norm is not None:  # QK-norm: normalise per head BEFORE RoPE
            q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rope(q, cos[start_pos:start_pos + T], sin[start_pos:start_pos + T])
        k = apply_rope(k, cos[start_pos:start_pos + T], sin[start_pos:start_pos + T])
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, heads, T, D]
        if cache is not None:
            k, v = cache.update(k, v, start_pos)
        mask, is_causal = _attention_mask(T, k.size(2), start_pos, x.device)
        dropout_p = self.cfg.dropout if self.training else 0.0
        if _SDPA_HAS_GQA:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal,
                                               dropout_p=dropout_p, enable_gqa=(H != Hkv))
        else:  # older PyTorch: materialise the shared KV heads
            rep = H // Hkv
            y = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1),
                                               attn_mask=mask, is_causal=is_causal, dropout_p=dropout_p)
        y = y.transpose(1, 2).reshape(B, T, H * D)
        if self.gate_proj is not None:  # head-specific elementwise sigmoid gate (Qwen "G1")
            y = y * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoE(nn.Module):
    """Token-choice top-k Mixture of Experts in the DeepSeek-V3 style.

    * Router: sigmoid affinities s = sigmoid(x W_r). Selection uses s + b (a per-expert bias
      that is NOT trained by gradients); combine weights use the un-biased s, renormalised
      over the selected experts.
    * Auxiliary-loss-free balancing (Wang et al. arXiv:2408.15664): after each optimizer
      step, b_e += rate * sign(mean_load - load_e). Overloaded experts become less likely to
      be picked without adding a loss term that fights the LM objective.
    * Optional classic Switch-style load-balancing loss (``moe_aux_loss_coef``).
    * Shared expert(s) always run: they capture common knowledge so routed experts specialise.

    The per-expert Python loop is for clarity. Real systems use grouped GEMMs
    (MegaBlocks / DeepGEMM) and expert parallelism with all-to-all token dispatch.
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(cfg.d_model, cfg.moe_d_ff) for _ in range(cfg.n_experts)])
        self.shared = (SwiGLU(cfg.d_model, cfg.moe_d_ff * cfg.n_shared_experts)
                       if cfg.n_shared_experts > 0 else None)
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts))
        self.register_buffer("load", torch.zeros(cfg.n_experts), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        E, k = self.cfg.n_experts, self.cfg.moe_top_k
        flat = x.reshape(-1, D)
        scores = torch.sigmoid(self.router(flat).float())                 # [N, E]
        _, idx = torch.topk(scores + self.expert_bias, k, dim=-1)         # [N, k]
        weights = scores.gather(-1, idx)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token_idx, slot = torch.where(idx == e)
            if token_idx.numel() == 0:
                continue
            y = expert(flat[token_idx]) * weights[token_idx, slot].unsqueeze(-1)
            out.index_add_(0, token_idx, y.to(out.dtype))
        if self.shared is not None:
            out = out + self.shared(flat).to(out.dtype)

        aux = x.new_zeros((), dtype=torch.float32)
        if self.training:
            counts = torch.bincount(idx.flatten(), minlength=E).float()
            self.load += counts.detach()
            if self.cfg.moe_aux_loss_coef > 0:
                frac_tokens = counts / (flat.size(0) * k)                     # f_e
                mean_prob = (scores / scores.sum(-1, keepdim=True)).mean(0)   # P_e
                aux = self.cfg.moe_aux_loss_coef * E * (frac_tokens * mean_prob).sum()
        return out.view(B, T, D), aux

    @torch.no_grad()
    def update_bias(self, rate: float) -> None:
        load = self.load.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load)  # keep the bias identical on every rank
        if load.sum() > 0:
            self.expert_bias += rate * torch.sign(load.mean() - load)
        self.load.zero_()


class Block(nn.Module):
    def __init__(self, cfg: LLMConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.is_moe = cfg.n_experts > 0 and layer_idx >= cfg.moe_first_dense
        self.mlp = MoE(cfg) if self.is_moe else SwiGLU(cfg.d_model, cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None, start_pos=0):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, cache, start_pos))
        h = self.mlp_norm(x)
        if self.is_moe:
            y, aux = self.mlp(h)
        else:
            y, aux = self.mlp(h), x.new_zeros((), dtype=torch.float32)
        return x + self.drop(y), aux


# --------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # Residual-branch output projections: shrink by 1/sqrt(2L) so the residual stream's
        # variance does not grow with depth at init (GPT-2 / Megatron convention).
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    # ---- forward ------------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                caches: list[KVCache] | None = None, start_pos: int = 0, last_only: bool = False) -> dict:
        B, T = idx.shape
        assert start_pos + T <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        x = self.embed(idx)
        aux_total = x.new_zeros((), dtype=torch.float32)
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            if self.cfg.grad_checkpointing and self.training and cache is None:
                # Activation checkpointing: drop block activations, recompute them in backward.
                # ~33% more compute for O(n_layers) less activation memory.
                x, aux = checkpoint(block, x, self.rope_cos, self.rope_sin, None, start_pos, use_reentrant=False)
            else:
                x, aux = block(x, self.rope_cos, self.rope_sin, cache, start_pos)
            aux_total = aux_total + aux
        x = self.norm(x)
        if last_only:
            x = x[:, -1:]
        logits = self.lm_head(x)
        if self.cfg.logit_softcap > 0:
            c = self.cfg.logit_softcap
            logits = c * torch.tanh(logits / c)
        out = {"logits": logits}
        if targets is not None:
            ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            loss = ce + aux_total
            zl = torch.zeros((), device=ce.device)
            if self.cfg.z_loss_coef > 0:
                zl = _z_loss(logits)
                loss = loss + self.cfg.z_loss_coef * zl
            out.update(loss=loss, ce=ce.detach(), z_loss=zl.detach(), aux_loss=aux_total.detach())
        return out

    # ---- utilities ----------------------------------------------------------------------
    def make_kv_cache(self, batch_size: int, max_len: int | None = None, device=None, dtype=None) -> list[KVCache]:
        device = device or self.embed.weight.device
        dtype = dtype or self.embed.weight.dtype
        max_len = max_len or self.cfg.max_seq_len
        return [KVCache(batch_size, self.cfg.n_kv_heads, max_len, self.cfg.head_dim, device, dtype)
                for _ in range(self.cfg.n_layers)]

    @torch.no_grad()
    def update_moe_biases(self) -> None:
        for m in self.modules():
            if isinstance(m, MoE):
                m.update_bias(self.cfg.moe_bias_update_rate)

    def moe_load_stats(self) -> dict[str, float]:
        """Max/mean expert load ratio per MoE layer (1.0 = perfectly balanced)."""
        stats = {}
        for name, m in self.named_modules():
            if isinstance(m, MoE) and m.load.sum() > 0:
                stats[name] = (m.load.max() / m.load.mean()).item()
        return stats

    def num_params(self, active_only: bool = False, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        moe_blocks = [b for b in self.blocks if b.is_moe]
        if active_only and moe_blocks:
            per_expert = sum(p.numel() for p in moe_blocks[0].mlp.experts[0].parameters())
            n -= len(moe_blocks) * (self.cfg.n_experts - self.cfg.moe_top_k) * per_expert
        return n

    def flops_per_token(self, seq_len: int) -> float:
        """Training FLOPs/token ~ 6*N_active + 12*L*H*D*T (PaLM appendix B). Used for MFU."""
        n = self.num_params(active_only=True, exclude_embedding=True) + self.cfg.d_model * self.cfg.vocab_size
        attn = 12 * self.cfg.n_layers * self.cfg.n_heads * self.cfg.head_dim * seq_len
        return 6 * n + attn
