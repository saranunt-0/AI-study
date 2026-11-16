"""Flow matching / rectified flow: training objective and ODE samplers.

The modern formulation (Lipman et al. arXiv:2210.02747; Liu et al. arXiv:2209.03003; used
by SD3 arXiv:2403.03206, FLUX, Movie Gen, Cosmos):

    x_t = (1 - t) * x_0 + t * eps,        t in [0, 1],  x_0 ~ data,  eps ~ N(0, I)
    target velocity   v = dx_t/dt = eps - x_0
    loss              E || v_theta(x_t, t, y) - v ||^2
    sampling          integrate dx/dt = v_theta from t = 1 (noise) to t = 0 (data)

Relation to DDPM / score models: it is the same family (Gaussian probability paths); a
straight-line path + velocity parameterisation gives straighter ODE trajectories, so fewer
sampling steps, and a simpler loss. See docs/03-diffusion-models.md for the full map.

Options implemented
* prediction="v"  : network outputs the velocity directly (SD3 default).
* prediction="x"  : network outputs the clean image x_0 and we convert v = (x_t - x̂_0) / t
                    (JiT, "Back to Basics", Li & He arXiv:2511.13720). Predicting the clean,
                    low-dimensional data manifold is easier for high-dim pixel inputs.
* t_sampling      : "uniform" or "logit_normal" (SD3): more training signal at mid-noise levels.
* samplers        : Euler (1st order) and Heun (2nd order) with an SD3-style time shift.
* classifier-free guidance (Ho & Salimans arXiv:2207.12598): v = v_u + w (v_c - v_u).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FlowConfig:
    prediction: str = "v"            # "v" or "x"
    t_sampling: str = "logit_normal"  # "logit_normal" or "uniform"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    t_min: float = 0.05              # clamp for the x->v conversion (avoid division by ~0)


class FlowMatching:
    def __init__(self, cfg: FlowConfig | None = None):
        self.cfg = cfg or FlowConfig()

    # ---- training -----------------------------------------------------------------------
    def sample_t(self, n: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.cfg.t_sampling == "uniform":
            return torch.rand(n, generator=generator).to(device)
        u = self.cfg.logit_mean + self.cfg.logit_std * torch.randn(n, generator=generator)
        return torch.sigmoid(u).to(device)

    def model_velocity(self, model, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = model(x_t, t, y)
        if self.cfg.prediction == "v":
            return out
        if self.cfg.prediction == "x":
            return (x_t - out) / t.clamp_min(self.cfg.t_min).view(-1, 1, 1, 1)
        raise ValueError(self.cfg.prediction)

    def loss(self, model, x0: torch.Tensor, y: torch.Tensor, generator: torch.Generator | None = None) -> dict:
        t = self.sample_t(x0.size(0), x0.device, generator)
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        tb = t.view(-1, 1, 1, 1)
        x_t = (1 - tb) * x0 + tb * noise
        v_target = noise - x0
        v_pred = self.model_velocity(model, x_t, t, y)
        loss = F.mse_loss(v_pred, v_target)
        return {"loss": loss}

    # ---- sampling -----------------------------------------------------------------------
    def guided_velocity(self, model, x, t: float, y, cfg_scale: float) -> torch.Tensor:
        tt = torch.full((x.size(0),), t, device=x.device)
        if cfg_scale == 1.0:
            return self.model_velocity(model, x, tt, y)
        null = torch.full_like(y, model.null_class)
        v = self.model_velocity(model, torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, null]))
        v_cond, v_uncond = v.chunk(2)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    @staticmethod
    def timesteps(steps: int, shift: float = 1.0, device=None) -> torch.Tensor:
        """t from 1 -> 0. shift > 1 spends more steps at high noise (SD3 uses 3.0 at 1024px)."""
        t = torch.linspace(1.0, 0.0, steps + 1, device=device)
        return shift * t / (1 + (shift - 1) * t)

    @torch.no_grad()
    def sample(self, model, shape, y: torch.Tensor, steps: int = 32, cfg_scale: float = 3.0,
               sampler: str = "heun", shift: float = 1.0, generator: torch.Generator | None = None,
               return_trajectory: bool = False):
        model.eval()
        device = y.device
        x = torch.randn(shape, generator=generator).to(device)
        ts = self.timesteps(steps, shift, device).tolist()
        traj = [x]
        for t, t_next in zip(ts[:-1], ts[1:]):
            dt = t_next - t                                         # negative
            v = self.guided_velocity(model, x, t, y, cfg_scale)
            if sampler == "euler" or t_next == 0.0:
                x = x + dt * v
            elif sampler == "heun":                                  # 2nd order: average slopes
                x_euler = x + dt * v
                v_next = self.guided_velocity(model, x_euler, t_next, y, cfg_scale)
                x = x + dt * 0.5 * (v + v_next)
            else:
                raise ValueError(sampler)
            if return_trajectory:
                traj.append(x)
        return (x, traj) if return_trajectory else x
