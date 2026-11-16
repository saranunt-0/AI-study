"""Inference CLI for the trained LM.

    python -m aistudy.llm.generate --ckpt runs/llm/best.pt --prompt "## Training" --temperature 0.8 --top-p 0.95
    python -m aistudy.llm.generate --ckpt runs/llm/best.pt --benchmark          # KV cache vs. no cache
    python -m aistudy.llm.generate --ckpt runs/big/best.pt --draft-ckpt runs/llm/best.pt --gamma 4   # speculative
"""

from __future__ import annotations

import argparse
import time

import torch

from aistudy.common.utils import get_device, load_checkpoint
from aistudy.llm.data import ByteTokenizer
from aistudy.llm.model import LLMConfig, Transformer
from aistudy.llm.sampling import generate, speculative_generate


def load_model(path: str, device: torch.device) -> Transformer:
    ckpt = load_checkpoint(path, map_location=device)
    model = Transformer(LLMConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--draft-ckpt", default=None, help="small model for speculative decoding")
    p.add_argument("--gamma", type=int, default=4, help="draft tokens per speculative round")
    p.add_argument("--prompt", default="# ")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = get_device(args.device)
    tok = ByteTokenizer()
    model = load_model(args.ckpt, device)
    prompt = torch.tensor([tok.encode(args.prompt, add_bos=True)] * args.num_samples, device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    if args.benchmark:
        n = min(args.max_new_tokens, model.cfg.max_seq_len - prompt.size(1) - 1)
        for use_cache in (False, True):
            t0 = time.time()
            generate(model, prompt[:1], n, temperature=0.0, use_cache=use_cache)
            dt = time.time() - t0
            print(f"use_cache={use_cache!s:5}  {n / dt:8.1f} tok/s")
        return

    if args.draft_ckpt:
        draft = load_model(args.draft_ckpt, device)
        out, stats = speculative_generate(model, draft, prompt[:1], args.max_new_tokens, args.gamma,
                                          args.temperature, args.top_k, args.top_p, args.min_p, gen)
        print(tok.decode(out[0].tolist()))
        print(f"\n[speculative] {stats}")
        return

    out = generate(model, prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p,
                   args.min_p, eos_id=tok.EOS, use_cache=not args.no_cache, generator=gen)
    for i in range(out.size(0)):
        print(f"--- sample {i} ---\n{tok.decode(out[i].tolist())}\n")


if __name__ == "__main__":
    main()
