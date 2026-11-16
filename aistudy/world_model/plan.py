"""Inference for the world model: goal-conditioned planning with CEM + MPC.

    python -m aistudy.world_model.plan --ckpt runs/world_model/last.pt --episodes 20
    python -m aistudy.world_model.plan --ckpt runs/world_model/last.pt --cross-room   # goal in the other room

Loop (model-predictive control):
    1. encode the current observation  -> z_t ;  encode the goal image -> z_goal
    2. CEM searches action sequences whose *imagined* latent rollout ends near z_goal
    3. execute only the first action, observe, re-plan (warm-started with the shifted plan)
Baselines: random actions, and "greedy" = move straight toward the goal using the TRUE state
(an oracle that ignores the wall) - it fails across rooms unless lucky with the door.
"""

from __future__ import annotations

import argparse

import torch

from aistudy.common.utils import get_device, load_checkpoint, save_image_grid, set_seed
from aistudy.world_model.env import EnvConfig, TwoRoomEnv
from aistudy.world_model.model import WMConfig, WorldModel, cem_plan


def sample_start_goal(env: TwoRoomEnv, cross_room: bool, gen: torch.Generator):
    while True:
        s, g = env.sample_valid_positions(2, gen)
        far = (s - g).norm() > 0.35
        if far and (not cross_room or env.room(s) != env.room(g)):
            return s, g


@torch.no_grad()
def run_episode(policy: str, model, env, start, goal, args, gen):
    obs = env.reset(start[None])
    z_goal = model.encode(env.render(goal[None])) if model is not None else None
    plan_mean = None
    frames, success, steps = [obs[0]], False, args.max_steps
    for t in range(args.max_steps):
        if policy == "cem":
            seq = cem_plan(model, model.encode(obs), z_goal, args.horizon, args.population, args.elites,
                           args.iterations, init_mean=plan_mean, generator=gen)
            action = seq[0]
            plan_mean = torch.cat([seq[1:], torch.zeros_like(seq[:1])])  # warm start next MPC step
        elif policy == "random":
            action = 2 * torch.rand(2, generator=gen).to(env.device) - 1
        else:  # greedy oracle on the true state
            d = goal - env.pos[0]
            action = (d / (d.abs().max() + 1e-8)).clamp(-1, 1)
        obs = env.step(action[None])
        frames.append(obs[0])
        if (env.pos[0] - goal).norm() < args.success_radius:
            success, steps = True, t + 1
            break
    return success, steps, torch.stack(frames), (env.pos[0] - goal).norm().item()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--population", type=int, default=256)
    p.add_argument("--elites", type=int, default=32)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--success-radius", type=float, default=0.08)
    p.add_argument("--cross-room", action="store_true")
    p.add_argument("--baselines", action="store_true", help="also run random / greedy-oracle policies")
    p.add_argument("--out", default=None, help="PNG with the first episodes' trajectories")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model = WorldModel(WMConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    env = TwoRoomEnv(EnvConfig(**ckpt["env_config"]), device=device)

    policies = ["cem"] + (["greedy", "random"] if args.baselines else [])
    gen = torch.Generator().manual_seed(args.seed)
    tasks = [sample_start_goal(env, args.cross_room, gen) for _ in range(args.episodes)]
    strips = []
    for policy in policies:
        succ, final_d = [], []
        pgen = torch.Generator(device=device).manual_seed(args.seed)
        for i, (s, g) in enumerate(tasks):
            ok, steps, frames, d = run_episode(policy, model if policy == "cem" else None, env, s, g, args, pgen)
            succ.append(ok)
            final_d.append(d)
            if policy == "cem" and i < 4 and args.out:
                idx = torch.linspace(0, frames.size(0) - 1, 10).long()
                goal_img = env.render(g[None])[0]
                strips.append(torch.cat([frames[idx], goal_img[None]]))
        rate = sum(succ) / len(succ)
        print(f"{policy:>7}: success {rate:6.1%}  mean final distance {sum(final_d) / len(final_d):.3f}  "
              f"({'cross-room' if args.cross_room else 'any'} goals, {len(tasks)} episodes)")
    if strips:
        save_image_grid(torch.cat(strips), args.out, nrow=11, value_range=(0, 1), upscale=3)
        print(f"saved {args.out}  (each row: 10 frames of the episode, last column = goal)")


if __name__ == "__main__":
    main()
