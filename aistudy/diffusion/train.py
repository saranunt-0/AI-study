"""Train a class-conditional DiT with flow matching on the procedural shapes dataset.

    python -m aistudy.diffusion.train                        # ~5-8 min CPU, v-prediction
    python -m aistudy.diffusion.train --prediction x         # JiT-style x-prediction
    python -m aistudy.diffusion.sample --ckpt runs/diffusion/last.pt --cfg 3.0

Generative-model training specifics shown here (vs. the LLM trainer):
* EMA weights for sampling (decay 0.999 here; 0.9999 for long ImageNet runs)
* label dropout (inside the model) so ONE network learns p(x|y) and p(x) -> CFG
* logit-normal timestep sampling, gradient clipping, warmup, periodic sample grids
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch

from aistudy.common.ema import EMA
from aistudy.common.optim import build_optimizer, lr_schedule, set_lr
from aistudy.common.utils import JsonlLogger, autocast_context, get_device, save_checkpoint, save_image_grid, set_seed
from aistudy.diffusion.data import ShapesDataset
from aistudy.diffusion.flow import FlowConfig, FlowMatching
from aistudy.diffusion.model import DiT, DiTConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--prediction", choices=["v", "x"], default="v")
    p.add_argument("--t-sampling", choices=["logit_normal", "uniform"], default="logit_normal")
    p.add_argument("--class-dropout", type=float, default=0.1)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--sample-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/diffusion")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    dtype = args.dtype or ("bf16" if device.type == "cuda" else "fp32")
    data = ShapesDataset(args.img_size)
    cfg = DiTConfig(img_size=args.img_size, patch_size=args.patch_size, d_model=args.d_model,
                    n_layers=args.n_layers, n_heads=args.n_heads, num_classes=data.num_classes,
                    class_dropout=args.class_dropout)
    fcfg = FlowConfig(prediction=args.prediction, t_sampling=args.t_sampling)
    model = DiT(cfg).to(device)
    flow = FlowMatching(fcfg)
    ema = EMA(model, decay=args.ema, warmup_steps=1)
    opt = build_optimizer(model, args.optimizer, args.lr, args.weight_decay, betas=(0.9, 0.99))
    fwd_model = torch.compile(model) if args.compile else model
    ctx = autocast_context(device, dtype)
    logger = JsonlLogger(os.path.join(args.out_dir, "log.jsonl"))
    gen = torch.Generator().manual_seed(args.seed)
    print(f"device={device} dtype={dtype} params={sum(p.numel() for p in model.parameters()):,} "
          f"prediction={fcfg.prediction}")

    for step in range(args.steps):
        lr = set_lr(opt, lr_schedule(step, args.steps, args.warmup, "constant"))
        x0, y = data.sample(args.batch_size, device, gen)
        model.train()
        with ctx:
            loss = flow.loss(fwd_model, x0, y, gen)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        ema.update(model)
        if step % args.log_every == 0 or step == args.steps - 1:
            logger.log({"step": step, "loss": loss.item(), "grad_norm": gn.item(), "lr": lr})
        if (step > 0 and step % args.sample_every == 0) or step == args.steps - 1:
            y_show = torch.arange(data.num_classes, device=device).repeat_interleave(8)
            imgs = flow.sample(ema.shadow, (y_show.numel(), 3, args.img_size, args.img_size), y_show,
                               steps=24, cfg_scale=3.0, generator=torch.Generator().manual_seed(0))
            save_image_grid(imgs, os.path.join(args.out_dir, f"samples_{step:06d}.png"), nrow=8, upscale=3)
            save_checkpoint(os.path.join(args.out_dir, "last.pt"), model=model, ema=ema, optimizer=opt,
                            config=asdict(cfg), flow_config=asdict(fcfg), step=step)


if __name__ == "__main__":
    main()
