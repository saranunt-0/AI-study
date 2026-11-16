"""Action-conditioned latent world model (JEPA-style) + planning utilities.

                o_t ──Encoder──▶ z_t ─┐
                a_t ──────────────────┼─▶ causal Transformer predictor ──▶ ẑ_{t+1}  ≈  Encoder(o_{t+1})
         (history z_<t, a_<t) ────────┘
     SIGReg(z) keeps the embedding an isotropic Gaussian  -> no representation collapse
     Decoder(z.detach()) is a *probe* for visualisation only (does not shape z)

This is the recipe of LeWorldModel (Maes, Le Lidec, Scieur, LeCun, Balestriero 2026,
arXiv:2603.19312): end-to-end JEPA from pixels, next-embedding prediction + SIGReg from LeJEPA
(arXiv:2511.08544). It sits in the "latent predictive" family (V-JEPA 2-AC, DINO-WM, PLDM,
TD-MPC2) as opposed to "pixel generative" world models (Genie 3/4, Cosmos, GameNGen,
DIAMOND) and RL-centric RSSM models (DreamerV3, Dreamer 4). See docs/02-world-models.md.

``objective="recon"`` switches to a reconstruction-based model (decoder gradients shape z,
no SIGReg) - the classic World Models / Dreamer route - so you can compare both.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from aistudy.llm.model import Block, LLMConfig, RMSNorm, precompute_rope


@dataclass
class WMConfig:
    img_size: int = 32
    in_channels: int = 3
    action_dim: int = 2
    latent_dim: int = 32
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    max_len: int = 32
    objective: str = "jepa"        # "jepa" (prediction + SIGReg) or "recon" (prediction + pixel reconstruction)
    channels: tuple[int, int, int] = (16, 32, 64)
    sigreg_weight: float = 1.0     # LeWM's single loss hyper-parameter (lambda)
    sigreg_slices: int = 256
    stop_grad_target: bool = False  # LeWM trains end-to-end; set True for a BYOL/SimSiam-style target


# --------------------------------------------------------------------------------------
# SIGReg: Sketched Isotropic Gaussian Regulariser (LeJEPA)
# --------------------------------------------------------------------------------------
def sigreg(z: torch.Tensor, num_slices: int = 256, t_max: float = 3.0, num_points: int = 17,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """Push the distribution of embeddings z [N, K] toward N(0, I).

    Cramér-Wold: a distribution is N(0, I) iff every 1-D projection is N(0, 1). We project
    onto ``num_slices`` random unit directions and, for each, compute the Epps-Pulley
    statistic: a weighted L2 distance between the empirical characteristic function
    phi_N(t) = mean_n exp(i t x_n) and the Gaussian one exp(-t^2/2), integrated over t.
    Differentiable, O(N * slices), no negatives, no EMA teacher, no stop-gradient.
    """
    N, K = z.shape
    A = torch.randn(K, num_slices, device=z.device, generator=generator)
    A = A / A.norm(dim=0, keepdim=True)
    proj = z.float() @ A                                               # [N, M]
    t = torch.linspace(0.0, t_max, num_points, device=z.device)       # integrand is even in t
    xt = proj.unsqueeze(-1) * t                                       # [N, M, P]
    gauss_cf = torch.exp(-0.5 * t ** 2)
    err = (torch.cos(xt).mean(0) - gauss_cf) ** 2 + torch.sin(xt).mean(0) ** 2
    stat = 2.0 * torch.trapezoid(err * gauss_cf, t, dim=-1)           # Gaussian weight w(t)
    return stat.mean()


# --------------------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------------------
def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1), nn.GroupNorm(8, cout), nn.SiLU())


class Encoder(nn.Module):
    def __init__(self, cfg: WMConfig):
        super().__init__()
        c1, c2, c3 = cfg.channels
        self.net = nn.Sequential(_conv_block(cfg.in_channels, c1), _conv_block(c1, c2), _conv_block(c2, c3))
        feat = c3 * (cfg.img_size // 8) ** 2
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(feat, 256), nn.SiLU(), nn.Linear(256, cfg.latent_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        lead = obs.shape[:-3]
        z = self.proj(self.net(obs.reshape(-1, *obs.shape[-3:])))
        return z.reshape(*lead, -1)


class Decoder(nn.Module):
    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.s = cfg.img_size // 8
        c1, c2, c3 = cfg.channels
        self.c3 = c3
        self.fc = nn.Linear(cfg.latent_dim, c3 * self.s * self.s)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(c3, c2, 4, 2, 1), nn.GroupNorm(8, c2), nn.SiLU(),
            nn.ConvTranspose2d(c2, c1, 4, 2, 1), nn.GroupNorm(8, c1), nn.SiLU(),
            nn.ConvTranspose2d(c1, cfg.in_channels, 4, 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        lead = z.shape[:-1]
        x = self.fc(z.reshape(-1, z.shape[-1])).view(-1, self.c3, self.s, self.s)
        x = torch.sigmoid(self.net(x))
        return x.reshape(*lead, *x.shape[-3:])


class Predictor(nn.Module):
    """Causal Transformer over (z_t, a_t) tokens that outputs ẑ_{t+1} = z_t + f(history).

    Reuses the exact LLM block (RoPE, QK-norm, SwiGLU): a world model's dynamics model is
    "a language model over states and actions". Causality lets it condition on history,
    which matters under partial observability (velocity, occlusion).
    """

    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.cfg = cfg
        bcfg = LLMConfig(vocab_size=8, max_seq_len=cfg.max_len, d_model=cfg.d_model, n_layers=cfg.n_layers,
                         n_heads=cfg.n_heads, n_kv_heads=cfg.n_heads, attn_gate=False, z_loss_coef=0.0)
        self.inp = nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.d_model)
        self.blocks = nn.ModuleList([Block(bcfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.latent_dim)
        nn.init.zeros_(self.out.weight)  # start as the identity dynamics z_{t+1} = z_t
        nn.init.zeros_(self.out.bias)
        cos, sin = precompute_rope(bcfg.head_dim, cfg.max_len, 10000.0)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z [B, T, K], a [B, T, A] -> ẑ [B, T, K] where ẑ[:, t] predicts z[:, t+1]."""
        h = self.inp(torch.cat([z, a], dim=-1))
        for blk in self.blocks:
            h, _ = blk(h, self.rope_cos, self.rope_sin)
        return z + self.out(self.norm(h))


class WorldModel(nn.Module):
    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.predictor = Predictor(cfg)
        self.decoder = Decoder(cfg)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @torch.no_grad()
    def imagine(self, z0: torch.Tensor, actions: torch.Tensor, context_z: torch.Tensor | None = None,
                context_a: torch.Tensor | None = None) -> torch.Tensor:
        return self.rollout(z0, actions, context_z, context_a)

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor, context_z: torch.Tensor | None = None,
                context_a: torch.Tensor | None = None) -> torch.Tensor:
        """Open-loop autoregressive rollout in latent space.

        z0 [B, K], actions [B, H, A] -> predicted latents [B, H, K] for steps 1..H.
        Optional history (context_z [B, C, K], context_a [B, C, A]) precedes z0.
        """
        zs = [z0[:, None]] if context_z is None else [context_z, z0[:, None]]
        acts = [] if context_a is None else [context_a]
        preds = []
        for h in range(actions.size(1)):
            acts.append(actions[:, h:h + 1])
            z_seq = torch.cat(zs, 1)[:, -self.cfg.max_len:]
            a_seq = torch.cat(acts, 1)[:, -self.cfg.max_len:]
            nxt = self.predictor(z_seq, a_seq)[:, -1]
            preds.append(nxt)
            zs.append(nxt[:, None])
        return torch.stack(preds, 1)

    def loss(self, obs: torch.Tensor, actions: torch.Tensor, rollout_steps: int = 4,
             generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """obs [B, T+1, C, H, W], actions [B, T, A]."""
        cfg = self.cfg
        z = self.encode(obs)                                   # [B, T+1, K]
        target = z[:, 1:].detach() if cfg.stop_grad_target else z[:, 1:]
        # (1) teacher-forced next-embedding prediction
        pred = self.predictor(z[:, :-1], actions)
        l_tf = F.mse_loss(pred, target)
        # (2) multi-step open-loop rollout: trains the model on its own predictions,
        #     which is what planning uses (fights compounding error)
        H = min(rollout_steps, actions.size(1))
        l_roll = torch.zeros((), device=obs.device)
        if H > 1:
            roll = self.rollout(z[:, 0], actions[:, :H])
            l_roll = F.mse_loss(roll, target[:, :H])
        out = {"pred": l_tf, "rollout": l_roll}
        # (3) anti-collapse term. Decoding every 4th frame is enough and saves ~40% compute.
        z_dec, obs_dec = z[:, ::4], obs[:, ::4]
        if cfg.objective == "jepa":
            out["sigreg"] = sigreg(z.reshape(-1, z.size(-1)), cfg.sigreg_slices, generator=generator)
            recon = self.decode(z_dec.detach())                # probe: no gradient into the encoder
        elif cfg.objective == "recon":
            out["sigreg"] = torch.zeros((), device=obs.device)
            recon = self.decode(z_dec)                         # reconstruction shapes the latent
        else:
            raise ValueError(cfg.objective)
        out["recon"] = F.mse_loss(recon, obs_dec)
        out["loss"] = l_tf + l_roll + cfg.sigreg_weight * out["sigreg"] + out["recon"]
        return out


# --------------------------------------------------------------------------------------
# Planning: Cross-Entropy Method (CEM) model-predictive control in latent space
# --------------------------------------------------------------------------------------
@torch.no_grad()
def cem_plan(model: WorldModel, z0: torch.Tensor, z_goal: torch.Tensor, horizon: int = 10,
             population: int = 256, elites: int = 32, iterations: int = 4, action_dim: int = 2,
             init_mean: torch.Tensor | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """Return the best action sequence [horizon, A] minimising latent distance to z_goal.

    Cost = sum_t w_t ||ẑ_t - z_goal||^2 with weights increasing toward the end of the horizon.
    CEM is gradient-free and embarrassingly parallel - the standard planner in DINO-WM,
    V-JEPA 2-AC, PLDM and LeWM. Alternatives: MPPI (TD-MPC2), gradient-based shooting,
    or learning a policy inside the model ("imagination" RL, Dreamer).
    """
    device = z0.device
    mean = init_mean.clone() if init_mean is not None else torch.zeros(horizon, action_dim, device=device)
    std = torch.full((horizon, action_dim), 0.8, device=device)
    weights = torch.linspace(0.2, 1.0, horizon, device=device)
    best_seq, best_cost = mean, float("inf")
    for _ in range(iterations):
        noise = torch.randn(population, horizon, action_dim, device=device, generator=generator)
        cand = (mean + std * noise).clamp(-1, 1)
        cand[0] = mean.clamp(-1, 1)                        # always re-evaluate the current mean
        preds = model.rollout(z0.expand(population, -1), cand)  # [P, H, K]
        cost = (((preds - z_goal) ** 2).sum(-1) * weights).sum(-1)
        idx = torch.topk(cost, elites, largest=False).indices
        elite = cand[idx]
        mean, std = elite.mean(0), elite.std(0).clamp_min(0.05)
        if cost[idx[0]] < best_cost:
            best_cost, best_seq = cost[idx[0]].item(), cand[idx[0]]
    return best_seq
