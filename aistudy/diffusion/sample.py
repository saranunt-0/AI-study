"""Inference for the flow-matching DiT: sampling grids, guidance sweeps and a quality metric.

    python -m aistudy.diffusion.sample --ckpt runs/diffusion/last.pt --cfg 3.0 --steps 24 --out samples.png
    python -m aistudy.diffusion.sample --ckpt runs/diffusion/last.pt --eval --cfg-sweep 1 2 3 5
    python -m aistudy.diffusion.sample --ckpt runs/diffusion/last.pt --trajectory --out traj.png

``--eval`` trains a small classifier on real data, then reports the fraction of generated
samples classified as the requested class ("conditional accuracy"). It is the same idea as
Inception Score / FID: judge a generator through a pretrained recogniser. Watch the classic
trade-off: higher CFG -> higher accuracy (fidelity) but less diversity (we log pixel std).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import nn

from aistudy.common.ema import EMA
from aistudy.common.utils import get_device, load_checkpoint, save_image_grid, set_seed
from aistudy.diffusion.data import CLASSES, ShapesDataset
from aistudy.diffusion.flow import FlowConfig, FlowMatching
from aistudy.diffusion.model import DiT, DiTConfig


def load(path: str, device, use_ema: bool = True):
    ckpt = load_checkpoint(path, map_location=device)
    model = DiT(DiTConfig(**ckpt["config"])).to(device)
    if use_ema and "ema" in ckpt:
        ema = EMA(model)
        ema.load_state_dict(ckpt["ema"])
        model = ema.shadow.to(device)
    else:
        model.load_state_dict(ckpt["model"])
    return model.eval(), FlowMatching(FlowConfig(**ckpt["flow_config"]))


def train_probe_classifier(data: ShapesDataset, device, steps: int = 300) -> nn.Module:
    clf = nn.Sequential(
        nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
        nn.Conv2d(64, 64, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, data.num_classes),
    ).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(123)
    for _ in range(steps):
        x, y = data.sample(128, device, gen)
        loss = F.cross_entropy(clf(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, y = data.sample(1024, device, gen)
    acc = (clf(x).argmax(-1) == y).float().mean().item()
    print(f"probe classifier accuracy on real data: {acc:.3f}")
    return clf.eval()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-per-class", type=int, default=8)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--sampler", choices=["euler", "heun"], default="heun")
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--shift", type=float, default=1.0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--out", default="samples.png")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--cfg-sweep", type=float, nargs="*", default=None)
    p.add_argument("--trajectory", action="store_true", help="save the denoising trajectory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    model, flow = load(args.ckpt, device, use_ema=not args.no_ema)
    data = ShapesDataset(model.cfg.img_size)
    shape = lambda n: (n, model.cfg.in_channels, model.cfg.img_size, model.cfg.img_size)  # noqa: E731

    if args.trajectory:
        y = torch.arange(data.num_classes, device=device)
        _, traj = flow.sample(model, shape(len(y)), y, args.steps, args.cfg, args.sampler, args.shift,
                              torch.Generator().manual_seed(args.seed), return_trajectory=True)
        idx = torch.linspace(0, len(traj) - 1, 9).long().tolist()
        grid = torch.stack([traj[i] for i in idx], 1).reshape(-1, *traj[0].shape[1:])
        save_image_grid(grid.clamp(-1, 1), args.out, nrow=len(idx), upscale=3)
        print(f"saved {args.out}: rows = classes {CLASSES}, columns = t from 1 (noise) to 0 (data)")
        return

    y = torch.arange(data.num_classes, device=device).repeat_interleave(args.n_per_class)
    imgs = flow.sample(model, shape(len(y)), y, args.steps, args.cfg, args.sampler, args.shift,
                       torch.Generator().manual_seed(args.seed))
    save_image_grid(imgs, args.out, nrow=args.n_per_class, upscale=3)
    print(f"saved {args.out}: rows = {CLASSES}, cfg={args.cfg}, steps={args.steps}, sampler={args.sampler}")

    if args.eval:
        clf = train_probe_classifier(data, device)
        for w in args.cfg_sweep or [args.cfg]:
            y = torch.arange(data.num_classes, device=device).repeat_interleave(64)
            x = flow.sample(model, shape(len(y)), y, args.steps, w, args.sampler, args.shift,
                            torch.Generator().manual_seed(args.seed))
            with torch.no_grad():
                acc = (clf(x.clamp(-1, 1)).argmax(-1) == y).float().mean().item()
            print(f"cfg={w:4.1f}  conditional accuracy={acc:.3f}  pixel std across samples={x.std(0).mean():.3f}")


if __name__ == "__main__":
    main()
