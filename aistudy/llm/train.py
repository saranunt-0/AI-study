"""Pre-train the decoder-only Transformer.

Quick start (CPU, ~2-4 min, trains on this repo's markdown):
    python -m aistudy.llm.train --preset tiny --steps 300

Mixture of Experts + Muon:
    python -m aistudy.llm.train --preset tiny --n-experts 8 --moe-top-k 2 --optimizer muon

Multi-GPU / multi-process (DDP; gloo on CPU, NCCL on GPU):
    torchrun --nproc_per_node=8 -m aistudy.llm.train --preset base --data data/mycorpus --dtype bf16 --compile

The loop shows the full "production" skeleton in ~250 lines: LR warmup + cosine/WSD, gradient
accumulation with DDP no_sync, bf16/fp16 autocast (+GradScaler), global-norm clipping, AGC,
loss-spike skipping, gradient-noise-scale (critical batch size) estimation, MoE bias balancing,
MFU logging, periodic eval, atomic checkpoints and exact resume.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import time
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from aistudy.common.grad import GradNoiseScale, SpikeGuard, adaptive_grad_clip_, clip_grad_norm_, global_grad_norm
from aistudy.common.optim import build_optimizer, lr_schedule, set_lr
from aistudy.common.utils import (JsonlLogger, autocast_context, get_device, init_distributed, load_checkpoint,
                                  save_checkpoint, set_seed)
from aistudy.llm.data import ByteTokenizer, load_splits
from aistudy.llm.model import LLMConfig, Transformer
from aistudy.llm.sampling import generate

PRESETS = {
    # name: (d_model, n_layers, n_heads, n_kv_heads, seq_len)
    "tiny": (128, 4, 4, 2, 128),      # ~1M params, CPU-friendly
    "small": (384, 6, 6, 2, 256),     # ~10M params
    "base": (768, 12, 12, 4, 1024),   # ~85M params, single GPU
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--data", nargs="*", default=None, help="text files/globs or a dir with train.bin/val.bin")
    g.add_argument("--val-frac", type=float, default=0.05)
    g = p.add_argument_group("model")
    g.add_argument("--preset", choices=PRESETS, default="tiny")
    g.add_argument("--d-model", type=int)
    g.add_argument("--n-layers", type=int)
    g.add_argument("--n-heads", type=int)
    g.add_argument("--n-kv-heads", type=int)
    g.add_argument("--seq-len", type=int)
    g.add_argument("--n-experts", type=int, default=0)
    g.add_argument("--moe-top-k", type=int, default=2)
    g.add_argument("--moe-aux-loss", type=float, default=0.0)
    g.add_argument("--moe-bias-rate", type=float, default=1e-3, help="aux-loss-free balancing step size")
    g.add_argument("--no-qk-norm", action="store_true")
    g.add_argument("--no-attn-gate", action="store_true")
    g.add_argument("--logit-softcap", type=float, default=0.0)
    g.add_argument("--z-loss", type=float, default=1e-4)
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--grad-checkpointing", action="store_true")
    g = p.add_argument_group("optimisation")
    g.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    g.add_argument("--lr", type=float, default=3e-3)
    g.add_argument("--min-lr-ratio", type=float, default=0.1)
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.95)
    g.add_argument("--schedule", choices=["cosine", "wsd", "constant"], default="cosine")
    g.add_argument("--warmup", type=int, default=50)
    g.add_argument("--decay-frac", type=float, default=0.2, help="WSD: fraction of steps spent decaying")
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--agc", type=float, default=0.0, help="adaptive gradient clipping factor (0 = off)")
    g.add_argument("--spike-guard", action="store_true", help="skip steps with outlier loss/grad-norm")
    g.add_argument("--noise-scale", action="store_true", help="estimate critical batch size (needs grad-accum>1)")
    g = p.add_argument_group("batch")
    g.add_argument("--batch-size", type=int, default=16, help="micro-batch size per process")
    g.add_argument("--grad-accum", type=int, default=1)
    g.add_argument("--steps", type=int, default=300)
    g = p.add_argument_group("system")
    g.add_argument("--device", default="auto")
    g.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default=None)
    g.add_argument("--compile", action="store_true")
    g.add_argument("--seed", type=int, default=1337)
    g.add_argument("--out-dir", default="runs/llm")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--eval-every", type=int, default=100)
    g.add_argument("--eval-iters", type=int, default=20)
    g.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    g.add_argument("--peak-tflops", type=float, default=0.0, help="per-device peak (e.g. 989 for H100 bf16) for MFU")
    g.add_argument("--sample-prompt", default="# ")
    return p.parse_args(argv)


def build_config(args, vocab_size: int) -> LLMConfig:
    d, L, H, Hkv, T = PRESETS[args.preset]
    return LLMConfig(
        vocab_size=8 * math.ceil(vocab_size / 8),
        max_seq_len=args.seq_len or T,
        d_model=args.d_model or d,
        n_layers=args.n_layers or L,
        n_heads=args.n_heads or H,
        n_kv_heads=args.n_kv_heads or Hkv,
        qk_norm=not args.no_qk_norm,
        attn_gate=not args.no_attn_gate,
        logit_softcap=args.logit_softcap,
        z_loss_coef=args.z_loss,
        dropout=args.dropout,
        n_experts=args.n_experts,
        moe_top_k=args.moe_top_k,
        moe_aux_loss_coef=args.moe_aux_loss,
        moe_bias_update_rate=args.moe_bias_rate,
        grad_checkpointing=args.grad_checkpointing,
    )


@torch.no_grad()
def evaluate(model, stream, args, cfg, device, ctx, generator) -> float:
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = stream.get_batch(args.batch_size, cfg.max_seq_len, device, generator)
        with ctx:
            losses.append(model(x, targets=y)["ce"].item())
    model.train()
    return sum(losses) / len(losses)


def main(argv=None) -> dict:
    args = parse_args(argv)
    is_ddp, rank, local_rank, world = init_distributed()
    master = rank == 0
    device = torch.device(f"cuda:{local_rank}") if (is_ddp and torch.cuda.is_available()) else get_device(args.device)
    dtype = args.dtype or ("bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32")
    set_seed(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for any fp32 matmul left over
        torch.backends.cudnn.allow_tf32 = True

    tok = ByteTokenizer()
    train_stream, val_stream = load_splits(args.data, args.val_frac)
    cfg = build_config(args, tok.vocab_size)
    model = Transformer(cfg).to(device)
    raw_model = model
    optimizer = build_optimizer(raw_model, args.optimizer, args.lr, args.weight_decay, (args.beta1, args.beta2))

    start_step, best_val = 0, float("inf")
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"] + 1, ckpt.get("best_val", best_val)
        if master:
            print(f"resumed from {args.resume} at step {start_step}")

    if args.compile:
        model = torch.compile(model)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    ctx = autocast_context(device, dtype)
    scaler = torch.amp.GradScaler(device.type, enabled=(dtype == "fp16"))
    spike_guard = SpikeGuard() if args.spike_guard else None
    noise_scale = GradNoiseScale() if (args.noise_scale and args.grad_accum > 1 and dtype != "fp16") else None
    logger = JsonlLogger(os.path.join(args.out_dir, "log.jsonl"), enabled=master)
    gen = torch.Generator().manual_seed(args.seed + 1000 * rank)
    eval_gen = torch.Generator().manual_seed(12345)

    tokens_per_micro = args.batch_size * cfg.max_seq_len
    tokens_per_step = tokens_per_micro * args.grad_accum * world
    flops_per_token = raw_model.flops_per_token(cfg.max_seq_len)
    if master:
        print(f"device={device} dtype={dtype} world={world} params={raw_model.num_params():,} "
              f"active={raw_model.num_params(active_only=True):,} tokens/step={tokens_per_step:,} "
              f"train_tokens={len(train_stream):,}")

    model.train()
    t_last = time.time()
    train_loss = float("nan")
    for step in range(start_step, args.steps):
        lr = set_lr(optimizer, lr_schedule(step, args.steps, args.warmup, args.schedule,
                                           args.min_lr_ratio, args.decay_frac))
        loss_sum = torch.zeros((), device=device)
        small_sq = None
        for micro in range(args.grad_accum):
            x, y = train_stream.get_batch(args.batch_size, cfg.max_seq_len, device, gen)
            last_micro = micro == args.grad_accum - 1
            # Skip the DDP all-reduce on all but the last micro-batch: gradients accumulate
            # locally and are averaged across ranks once per optimizer step.
            sync_ctx = model.no_sync() if (is_ddp and not last_micro) else contextlib.nullcontext()
            with sync_ctx:
                with ctx:
                    out = model(x, targets=y)
                    loss = out["loss"] / args.grad_accum
                scaler.scale(loss).backward()
            loss_sum += out["ce"].detach() / args.grad_accum
            if noise_scale is not None and micro == 0:
                small_sq = (global_grad_norm(raw_model.parameters()).item() * args.grad_accum) ** 2

        if is_ddp:  # identical loss on every rank -> identical skip decisions
            dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        train_loss = loss_sum.item()

        scaler.unscale_(optimizer)  # grads are now in true units (no-op unless fp16)
        grad_norm = clip_grad_norm_(raw_model.parameters(), 0.0).item()  # pre-clip norm
        ns = None
        if noise_scale is not None and small_sq is not None:
            ns = noise_scale.update(small_sq, grad_norm ** 2, tokens_per_micro, tokens_per_step)
        if args.agc > 0:
            adaptive_grad_clip_(raw_model.parameters(), args.agc)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

        skip = (spike_guard.should_skip(train_loss, grad_norm) if spike_guard
                else not (math.isfinite(train_loss) and math.isfinite(grad_norm)))
        if skip:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            if master:
                print(f"step {step}: skipped update (loss={train_loss:.3f}, grad_norm={grad_norm:.3g})")
            continue

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        moe_stats = raw_model.moe_load_stats()  # read expert loads before the bias update resets them
        raw_model.update_moe_biases()

        if step % args.log_every == 0 or step == args.steps - 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t_last) / (args.log_every if step > start_step else 1)
            t_last = time.time()
            rec = {"step": step, "loss": train_loss, "lr": lr, "grad_norm": grad_norm,
                   "tok_per_s": tokens_per_step / max(dt, 1e-9)}
            if args.peak_tflops > 0:
                rec["mfu"] = flops_per_token * rec["tok_per_s"] / (args.peak_tflops * 1e12 * world)
            if ns is not None:
                rec["noise_scale_tokens"] = ns
            if moe_stats:
                rec["moe_max_load_ratio"] = max(moe_stats.values())
            logger.log(rec)

        last = step == args.steps - 1
        if step > 0 and (step % args.eval_every == 0 or last):
            val_loss = evaluate(raw_model, val_stream, args, cfg, device, ctx, eval_gen)
            if is_ddp:
                t = torch.tensor(val_loss, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                val_loss = t.item()
            if master:
                logger.log({"step": step, "val_loss": val_loss, "val_bpb": val_loss / math.log(2)})
                state = dict(model=raw_model, optimizer=optimizer, step=step, config=asdict(cfg),
                             args=vars(args), best_val=min(best_val, val_loss))
                save_checkpoint(os.path.join(args.out_dir, "last.pt"), **state)
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(os.path.join(args.out_dir, "best.pt"), **state)

    result = {"final_train_loss": train_loss, "best_val_loss": best_val}
    if master:
        prompt = torch.tensor([tok.encode(args.sample_prompt, add_bos=True)], device=device)
        out = generate(raw_model, prompt, max_new_tokens=200, temperature=0.8, top_p=0.95,
                       generator=torch.Generator(device=device).manual_seed(0) if device.type == "cuda"
                       else torch.Generator().manual_seed(0))
        print("\n--- sample ---\n" + tok.decode(out[0].tolist()) + "\n--------------")
    if is_ddp:
        dist.destroy_process_group()
    return result


if __name__ == "__main__":
    main()
