"""Diffusion Transformer (DiT) with adaLN-Zero conditioning.

DiT (Peebles & Xie 2023, arXiv:2212.09748) replaced the U-Net as the backbone of modern
diffusion / flow models (SD3, FLUX, Sora-style video models, Movie Gen, Cosmos...).

    image ─patchify─▶ tokens + 2D sincos pos ─▶ N x DiTBlock ─▶ adaLN final layer ─▶ unpatchify
    timestep t ─sinusoidal─▶ MLP ─┐
    class y (or "null") ─embed────┴─▶ c ─▶ per-block (shift, scale, gate) x 2

adaLN-Zero: the conditioning vector c regresses shift/scale/gate for the two LayerNorms of each
block; the gate-producing layer is ZERO-initialised so every residual block starts as the
identity - a key trick for stable training of deep conditional Transformers.

Modern additions used here: QK-norm in attention (SD3, FLUX) and SwiGLU MLP (LightningDiT).
Not included (see docs/03-diffusion-models.md): MM-DiT joint text-image attention (SD3),
RoPE-2D/3D for variable resolution & video, latent space (VAE / RAE) instead of pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from aistudy.llm.model import RMSNorm, SwiGLU


@dataclass
class DiTConfig:
    img_size: int = 32
    in_channels: int = 3
    patch_size: int = 4
    d_model: int = 192
    n_layers: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    num_classes: int = 4
    class_dropout: float = 0.1   # label dropout -> enables classifier-free guidance
    qk_norm: bool = True

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2


def sincos_2d(d_model: int, grid: int) -> torch.Tensor:
    """Fixed 2-D sine-cosine position embedding [grid*grid, d_model] (MAE / DiT)."""
    assert d_model % 4 == 0
    coords = torch.arange(grid, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    omega = 1.0 / 10000 ** (torch.arange(d_model // 4, dtype=torch.float32) / (d_model // 4))

    def emb(pos):
        out = pos.flatten()[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)

    return torch.cat([emb(yy), emb(xx)], dim=1)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of continuous t in [0, 1] (scaled by 1000 like discrete DDPM steps)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = 1000.0 * t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class SelfAttention(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.n_heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(y.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d = cfg.d_model
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(cfg)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(d, int(cfg.mlp_ratio * d * 2 / 3))  # 2/3 keeps params equal to a GELU MLP
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate_a[:, None] * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m[:, None] * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class FinalLayer(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d = cfg.d_model
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.linear = nn.Linear(d, cfg.patch_size ** 2 * cfg.in_channels)

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.x_embed = nn.Conv2d(cfg.in_channels, d, cfg.patch_size, cfg.patch_size)  # patchify
        self.register_buffer("pos_embed", sincos_2d(d, cfg.img_size // cfg.patch_size)[None], persistent=False)
        self.t_embed = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.y_embed = nn.Embedding(cfg.num_classes + 1, d)  # last index = "null" (unconditional)
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_layer = FinalLayer(cfg)
        self._init_weights()

    @property
    def null_class(self) -> int:
        return self.cfg.num_classes

    def _init_weights(self):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(basic)
        w = self.x_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))
        nn.init.zeros_(self.x_embed.bias)
        nn.init.normal_(self.y_embed.weight, std=0.02)
        nn.init.normal_(self.t_embed[0].weight, std=0.02)
        nn.init.normal_(self.t_embed[2].weight, std=0.02)
        for blk in self.blocks:  # adaLN-Zero
            nn.init.zeros_(blk.adaLN[-1].weight)
            nn.init.zeros_(blk.adaLN[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p, c = self.cfg.patch_size, self.cfg.in_channels
        h = w = self.cfg.img_size // p
        x = x.view(x.size(0), h, w, p, p, c)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(x.size(0), c, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x [B, C, H, W] noisy image, t [B] in [0, 1], y [B] class ids (null_class = unconditional)."""
        if self.training and self.cfg.class_dropout > 0:
            drop = torch.rand(y.shape, device=y.device) < self.cfg.class_dropout
            y = torch.where(drop, torch.full_like(y, self.null_class), y)
        h = self.x_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.t_embed(timestep_embedding(t, 256)) + self.y_embed(y)
        for blk in self.blocks:
            h = blk(h, c)
        return self.unpatchify(self.final_layer(h, c))
