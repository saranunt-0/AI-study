"""Train the latent world model on random-exploration data from TwoRoom.

    python -m aistudy.world_model.train                       # JEPA + SIGReg (default), ~5 min CPU
    python -m aistudy.world_model.train --objective recon     # reconstruction baseline
    python -m aistudy.world_model.plan --ckpt runs/world_model/last.pt    # then plan with it

Diagnostics logged during training (these are how you know a world model is any good):
* pred / rollout   : 1-step and multi-step latent prediction error
* sigreg           : distance of the embedding distribution from N(0, I)
* probe_r2         : linear-probe R^2 of the true (x, y) position from z  (is the state in z?)
* rollout_nmse@h   : open-loop error after h steps, normalised by latent variance (should stay << 1)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch

from aistudy.common.optim import build_optimizer, lr_schedule, set_lr
from aistudy.common.utils import JsonlLogger, get_device, save_checkpoint, save_image_grid, set_seed
from aistudy.world_model.env import EnvConfig, TwoRoomEnv, collect_random_trajectories
from aistudy.world_model.model import WMConfig, WorldModel


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objective", choices=["jepa", "recon"], default="jepa")
    p.add_argument("--sigreg-weight", type=float, default=1.0)
    p.add_argument("--stop-grad-target", action="store_true")
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-traj", type=int, default=4000)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=8)
    p.add_argument("--no-wall", action="store_true")
    p.add_argument("--obs-noise", type=float, default=0.02, help="pixel noise augmentation")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/world_model")
    return p.parse_args(argv)


def make_batch(env, data, idx, noise: float, generator=None):
    states, actions = data["states"][idx], data["actions"][idx]
    obs = env.render(states)
    if noise > 0:
        obs = (obs + noise * torch.randn(obs.shape, generator=generator).to(obs.device)).clamp(0, 1)
    return obs, actions, states


@torch.no_grad()
def evaluate(model, env, train_data, val_data, args, device, step, out_dir):
    model.eval()
    gen = torch.Generator().manual_seed(0)
    # --- linear probe: is the true position linearly decodable from z? ---
    def latents(data, n):
        idx = torch.randint(0, data["states"].size(0), (n,), generator=gen)
        obs, _, states = make_batch(env, data, idx, 0.0)
        return model.encode(obs[:, 0]), states[:, 0]
    z_tr, s_tr = latents(train_data, 2048)
    z_va, s_va = latents(val_data, 1024)
    X_tr = torch.cat([z_tr, torch.ones(len(z_tr), 1, device=device)], 1)
    X_va = torch.cat([z_va, torch.ones(len(z_va), 1, device=device)], 1)
    W = torch.linalg.lstsq(X_tr.cpu(), s_tr.cpu()).solution.to(device)
    resid = ((X_va @ W - s_va) ** 2).sum(0)
    total = ((s_va - s_va.mean(0)) ** 2).sum(0)
    probe_r2 = (1 - resid / total).mean().item()
    # --- open-loop rollout error vs horizon ---
    idx = torch.arange(min(256, val_data["states"].size(0)))
    obs, actions, _ = make_batch(env, val_data, idx, 0.0)
    z = model.encode(obs)
    roll = model.rollout(z[:, 0], actions)
    var = z.reshape(-1, z.size(-1)).var(0).mean()
    nmse = ((roll - z[:, 1:]) ** 2).mean(dim=(0, 2)) / var
    rec = {"step": step, "probe_r2": probe_r2}
    for h in (1, 4, 8, args.horizon):
        if h <= nmse.numel():
            rec[f"rollout_nmse@{h}"] = nmse[h - 1].item()
    # --- picture: top = real frames, bottom = decoded imagination ---
    n_show = 4
    frames = []
    for i in range(n_show):
        frames.append(obs[i, 1:13])
        frames.append(model.decode(roll[i, :12]))
    save_image_grid(torch.cat(frames), os.path.join(out_dir, f"imagination_{step:06d}.png"),
                    nrow=12, value_range=(0, 1), upscale=2)
    model.train()
    return rec


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    env = TwoRoomEnv(EnvConfig(wall=not args.no_wall), device=device)
    gen = torch.Generator().manual_seed(args.seed)
    train_data = collect_random_trajectories(env, args.n_traj, args.horizon, gen)
    val_data = collect_random_trajectories(env, max(256, args.n_traj // 10), args.horizon, gen)

    cfg = WMConfig(latent_dim=args.latent_dim, d_model=args.d_model, n_layers=args.n_layers,
                   max_len=max(32, args.horizon + 1), objective=args.objective,
                   sigreg_weight=args.sigreg_weight, stop_grad_target=args.stop_grad_target)
    model = WorldModel(cfg).to(device)
    opt = build_optimizer(model, "adamw", args.lr, args.weight_decay, betas=(0.9, 0.99))
    logger = JsonlLogger(os.path.join(args.out_dir, "log.jsonl"))
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,} objective={cfg.objective}")

    sig_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    for step in range(args.steps):
        lr = set_lr(opt, lr_schedule(step, args.steps, args.warmup, "cosine", 0.05))
        idx = torch.randint(0, args.n_traj, (args.batch_size,), generator=gen)
        obs, actions, _ = make_batch(env, train_data, idx, args.obs_noise, gen)
        out = model.loss(obs, actions, args.rollout_steps, generator=sig_gen)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            logger.log({"step": step, "lr": lr, "grad_norm": gn.item(),
                        **{k: v.item() for k, v in out.items()}})
        if (step > 0 and step % args.eval_every == 0) or step == args.steps - 1:
            logger.log(evaluate(model, env, train_data, val_data, args, device, step, args.out_dir))
            save_checkpoint(os.path.join(args.out_dir, "last.pt"), model=model, config=asdict(cfg),
                            env_config=asdict(env.cfg), step=step)


if __name__ == "__main__":
    main()
