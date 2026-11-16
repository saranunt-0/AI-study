# 07 · Inference and Serving

> **Code:** [`aistudy/llm/sampling.py`](../aistudy/llm/sampling.py) (KV-cached generation, temperature / top-k / top-p / min-p, speculative decoding) · [`aistudy/llm/model.py`](../aistudy/llm/model.py) (`KVCache`, chunked prefill masks) · [`aistudy/diffusion/flow.py`](../aistudy/diffusion/flow.py) (ODE samplers, CFG) · [`aistudy/world_model/model.py`](../aistudy/world_model/model.py) (`cem_plan`)

## Short conclusion

- **LLM decoding is memory-bandwidth bound.** Each new token reads all active weights plus the whole KV cache. Every serving technique attacks this:
  - **batching** (continuous batching) amortises weight reads;
  - **KV-cache management** (PagedAttention, prefix caching, GQA/MLA/sparse attention, KV quantisation) shrinks or reuses cache reads;
  - **speculative decoding** (EAGLE-3, MTP heads, draft models) produces several tokens per weight read;
  - **quantisation** (FP8, INT4/NVFP4 weights) shrinks the bytes.
- **Prefill is compute-bound; decode is bandwidth-bound.** Large deployments separate them onto different GPU pools (prefill/decode disaggregation).
- **Use a production engine** (vLLM, SGLang, TensorRT-LLM, llama.cpp for edge). Learn the internals from small code like this repo's, but don't rebuild the engine.

---

## 1. The KV cache (implemented here)

- **Prefill:** process the prompt in one parallel pass and store K/V for every layer.
- **Decode:** feed one token at a time, attend to the cache, and append.
- **Without a cache**, step t recomputes all t positions: O(T²) work per sequence and O(T³) for the whole generation. With a cache it is O(T) per step.

In this repo:
- `test_kv_cache_matches_full_forward` checks that prefill, *chunked* prefill and single-token decoding reproduce the full forward pass to 1e-4;
- `python -m aistudy.llm.generate --benchmark` measures the speed-up.

## 2. Techniques

| Technique | Idea | Link |
|---|---|---|
| Continuous batching | Add or remove sequences every step instead of waiting for the batch to finish | [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) |
| PagedAttention | Allocate the KV cache in fixed-size blocks (like OS pages) → no fragmentation, sharing | [vLLM paper](https://arxiv.org/abs/2309.06180), [vLLM](https://github.com/vllm-project/vllm) |
| Prefix / radix caching | Reuse the KV cache of shared prompt prefixes (system prompts, few-shot, multi-turn) | [SGLang](https://arxiv.org/abs/2312.07104), [repo](https://github.com/sgl-project/sglang) |
| Chunked prefill | Split long prompts into chunks interleaved with decode steps | (supported by `_attention_mask` here) |
| Prefill/decode disaggregation | Separate GPU pools for the compute-bound and bandwidth-bound phases | [DistServe](https://arxiv.org/abs/2401.09670), [Mooncake](https://arxiv.org/abs/2407.00079) |
| **Speculative decoding** | Draft k tokens cheaply, verify in one target pass, accept/reject → exact target distribution | [Leviathan et al.](https://arxiv.org/abs/2211.17192), [Chen et al.](https://arxiv.org/abs/2302.01318) |
| Draft heads | Medusa heads; EAGLE feature-level drafting; DeepSeek MTP modules reused as drafters | [Medusa](https://arxiv.org/abs/2401.10774), [EAGLE-3](https://arxiv.org/abs/2503.01840) |
| Weight quantisation | GPTQ, AWQ (INT4); FP8; NVFP4 on Blackwell | [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SmoothQuant](https://arxiv.org/abs/2211.10438) |
| KV quantisation / eviction | FP8/INT4 KV; keep sinks + recent window (StreamingLLM) | [StreamingLLM](https://arxiv.org/abs/2309.17453) |
| Structured output | Constrain decoding with grammars/JSON schemas | [Outlines](https://github.com/dottxt-ai/outlines), [XGrammar](https://arxiv.org/abs/2411.15100) |

**Speculative decoding in this repo.** `speculative_generate` implements the exact rejection-sampling algorithm with KV-cache rollback for both models. The tests check two things:
- with greedy decoding, the output equals the target's own greedy output;
- a model drafting for itself gets 100% acceptance.

## 3. Sampling

| Method | Effect | When |
|---|---|---|
| Greedy (T = 0) | Deterministic, can loop | Short factual answers, evals |
| Temperature | Sharpens or flattens the distribution | Always combined with a truncation |
| Top-k | Keep k most likely | Simple; not adaptive |
| Top-p (nucleus) | Keep smallest set with mass ≥ p | Default 0.9–0.95 ([Holtzman et al.](https://arxiv.org/abs/1904.09751)) |
| Min-p | Keep tokens with p ≥ min_p · p_max | Robust at high temperature ([Nguyen et al.](https://arxiv.org/abs/2407.01082)) |
| Repetition / frequency penalties | Discourage loops | Chat UIs; harmful for code |

## 4. Diffusion and world-model inference

- **Diffusion:**
  - fewer steps with better solvers (Heun, DPM-Solver++) or distillation (consistency, MeanFlow);
  - CFG doubles the batch (done in one forward pass here) and can be limited to an interval of t;
  - cache features across steps;
  - quantise to FP8.
- **World models:** planning cost = population × horizon × iterations forward passes per action. Latent models are cheap for this reason (LeWM reports up to 48× faster planning than foundation-model world models). Warm-starting CEM with the shifted previous plan (done in `plan.py`) cuts iterations.

## 5. Learn more

- [vLLM docs: design](https://docs.vllm.ai/) and [SGLang docs](https://docs.sglang.ai/)
- [How to Scale Your Model: inference chapter](https://jax-ml.github.io/scaling-book/), with roofline analysis of prefill and decode
- [Lilian Weng, *Large Transformer Model Inference Optimization*](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
