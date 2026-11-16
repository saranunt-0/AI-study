"""TwoRoom: a tiny, fully vectorised pixel environment for world-model research.

A point agent moves in the unit square. A vertical wall splits it into two rooms connected by
a door. Observations are RGB images (agent = red/green blob, wall = blue). Actions are 2-D
velocities in [-1, 1]^2. This mirrors the "Two-Rooms" navigation task used to study latent
planning (PLDM, arXiv:2502.14819; LeWorldModel, arXiv:2603.19312): simple enough to train in
minutes on a CPU, but planning across the door requires the model to understand obstacles -
greedy "move toward the goal" fails when the goal is in the other room.

Everything is batched torch, so collecting 100k transitions takes well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EnvConfig:
    img_size: int = 32
    step_size: float = 0.08
    wall: bool = True
    wall_x: float = 0.5
    wall_half_width: float = 0.03
    door_low: float = 0.35
    door_high: float = 0.65
    agent_radius: float = 0.02      # collision radius
    blob_sigma: float = 0.05        # rendering width
    margin: float = 0.04            # keep the agent inside the image


class TwoRoomEnv:
    def __init__(self, cfg: EnvConfig | None = None, device: torch.device | str = "cpu"):
        self.cfg = cfg or EnvConfig()
        self.device = torch.device(device)
        n = self.cfg.img_size
        coords = (torch.arange(n, device=self.device, dtype=torch.float32) + 0.5) / n
        self.yy, self.xx = torch.meshgrid(coords, coords, indexing="ij")  # row = y, col = x
        self.wall_mask = self._wall_mask()
        self.pos: torch.Tensor | None = None

    # ---- geometry -----------------------------------------------------------------------
    def _wall_mask(self) -> torch.Tensor:
        c = self.cfg
        if not c.wall:
            return torch.zeros_like(self.xx)
        in_band = (self.xx - c.wall_x).abs() < c.wall_half_width
        in_door = (self.yy > c.door_low) & (self.yy < c.door_high)
        return (in_band & ~in_door).float()

    def valid(self, pos: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        if not c.wall:
            return torch.ones(pos.shape[:-1], dtype=torch.bool, device=pos.device)
        x, y = pos[..., 0], pos[..., 1]
        in_band = (x - c.wall_x).abs() < c.wall_half_width + c.agent_radius
        in_door = (y > c.door_low + c.agent_radius) & (y < c.door_high - c.agent_radius)
        return ~in_band | in_door

    def room(self, pos: torch.Tensor) -> torch.Tensor:
        """0 = left room, 1 = right room."""
        return (pos[..., 0] > self.cfg.wall_x).long()

    def sample_valid_positions(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        c = self.cfg
        out = torch.empty(0, 2, device=self.device)
        while out.size(0) < n:
            cand = c.margin + (1 - 2 * c.margin) * torch.rand(2 * n, 2, generator=generator).to(self.device)
            out = torch.cat([out, cand[self.valid(cand)]])
        return out[:n]

    # ---- dynamics -----------------------------------------------------------------------
    def dynamics(self, pos: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Move, clip to the box, and slide along walls instead of passing through them."""
        c = self.cfg
        delta = c.step_size * action.clamp(-1, 1)
        cand = (pos + delta).clamp(c.margin, 1 - c.margin)
        slide_y = torch.stack([pos[..., 0], cand[..., 1]], -1)   # keep x, move y
        slide_x = torch.stack([cand[..., 0], pos[..., 1]], -1)   # keep y, move x
        ok, ok_y, ok_x = self.valid(cand), self.valid(slide_y), self.valid(slide_x)
        new = torch.where(ok[..., None], cand,
                          torch.where(ok_y[..., None], slide_y, torch.where(ok_x[..., None], slide_x, pos)))
        return new

    def reset(self, pos: torch.Tensor) -> torch.Tensor:
        self.pos = pos.to(self.device).float()
        return self.render(self.pos)

    def step(self, action: torch.Tensor) -> torch.Tensor:
        self.pos = self.dynamics(self.pos, action.to(self.device))
        return self.render(self.pos)

    # ---- rendering ----------------------------------------------------------------------
    def render(self, pos: torch.Tensor) -> torch.Tensor:
        """pos [..., 2] -> images [..., 3, H, W] in [0, 1]."""
        c = self.cfg
        lead = pos.shape[:-1]
        p = pos.reshape(-1, 2)
        dx = self.xx[None] - p[:, 0, None, None]
        dy = self.yy[None] - p[:, 1, None, None]
        blob = torch.exp(-(dx ** 2 + dy ** 2) / (2 * c.blob_sigma ** 2))
        wall = self.wall_mask.expand_as(blob)
        img = torch.stack([blob, 0.6 * blob, wall], dim=1)  # R, G = agent; B = wall
        return img.reshape(*lead, 3, c.img_size, c.img_size)


def collect_random_trajectories(env: TwoRoomEnv, n_traj: int, horizon: int, generator: torch.Generator,
                                door_frac: float = 0.25) -> dict[str, torch.Tensor]:
    """Random exploration data: piecewise-persistent random actions.

    ``door_frac`` of the episodes start near the door. Data coverage is THE bottleneck of
    offline world models: a model can only plan through transitions it has seen.
    """
    c = env.cfg
    starts = env.sample_valid_positions(n_traj, generator)
    n_door = int(door_frac * n_traj)
    if c.wall and n_door > 0:
        door = torch.stack([c.wall_x + 0.2 * (torch.rand(n_door, generator=generator) - 0.5),
                            c.door_low + (c.door_high - c.door_low) * torch.rand(n_door, generator=generator)], -1)
        door = door.to(env.device)
        starts[:n_door] = torch.where(env.valid(door)[:, None], door, starts[:n_door])
    states = [starts]
    actions = []
    act = 2 * torch.rand(n_traj, 2, generator=generator) - 1
    for _ in range(horizon):
        resample = torch.rand(n_traj, 1, generator=generator) < 0.3      # keep direction ~3 steps
        act = torch.where(resample, 2 * torch.rand(n_traj, 2, generator=generator) - 1, act)
        a = act.to(env.device)
        actions.append(a)
        states.append(env.dynamics(states[-1], a))
    return {"states": torch.stack(states, 1), "actions": torch.stack(actions, 1)}
