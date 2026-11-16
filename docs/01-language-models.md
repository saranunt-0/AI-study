# 01 · Language Models: Architecture in 2026

> **Code:** [`aistudy/llm/`](../aistudy/llm): `model.py` (architecture) · `train.py` (pre-training) · `generate.py` / `sampling.py` (inference, KV cache, speculative decoding)

## Short conclusion

- **The core block has converged.** Almost every frontier open-weights LLM (DeepSeek, Qwen, Kimi, Llama, Gemma, gpt-oss, OLMo) is a decoder-only Transformer with **pre-norm RMSNorm, RoPE, SwiGLU, grouped-query or latent attention, and no biases**. QK-norm and attention-output gating are the 2025 additions.
- **Frontier models are sparse Mixtures of Experts (MoE).** Typically only 1–6% of parameters are active per token (DeepSeek-V4-Pro: 1.6T total / 49B active; Kimi K2: 1T / 32B; Qwen3.5: 397B / 17B). The 2024–25 trend was "fine-grained experts + a shared expert + auxiliary-loss-free load balancing".
- **2025–26 innovation is in attention efficiency for million-token contexts:**
  - compressed KV caches: MLA in DeepSeek-V2/V3;
  - learned sparse attention: NSA, DeepSeek Sparse Attention in V3.2, and the "compressed sparse" + "heavily compressed" hybrid in V4;
  - linear-attention hybrids: Gated DeltaNet in a 3:1 mix with full attention in Qwen3-Next and Qwen3.5, Kimi Linear, Mamba-hybrids.
- **The residual stream is being redesigned for the first time since 2016.** DeepSeek's manifold-constrained hyper-connections (mHC) are used in DeepSeek-V4.
- **Architecture is not where most of the gains come from.** Data quality, the scale of training compute, and post-training (RL) dominate. Copy a proven architecture, then spend your effort on data, training stability and evaluation. Exotic block designs are a research bet, not a product decision.

---

## 1. The canonical 2026 block

```
x ─┬─ RMSNorm ─ Attention (GQA/MLA, RoPE, QK-norm, sigmoid output gate) ─(+)─┬─ RMSNorm ─ FFN (SwiGLU or MoE) ─(+)─▶
   └─────────────────────────────────────────────────────────────────────────┘ └─────────────────────────────────┘
```

| Component | What it does | Why it won | Primary source |
|---|---|---|---|
| Decoder-only, causal LM | One objective (next-token prediction) for everything | Simplest to scale; emergent in-context learning | [Radford et al. 2019 (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [Brown et al. 2020 (GPT-3)](https://arxiv.org/abs/2005.14165) |
| Pre-norm + RMSNorm | Normalises inputs of each sub-layer; residual path stays an identity | Stable at depth; RMSNorm is cheaper than LayerNorm and equally good | [Xiong et al. 2020](https://arxiv.org/abs/2002.04745), [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467) |
| RoPE | Rotates q/k by position-dependent angles → relative positions | Works with KV cache, extendable (NTK/YaRN) | [Su et al. 2021](https://arxiv.org/abs/2104.09864), [YaRN](https://arxiv.org/abs/2309.00071) |
| SwiGLU FFN | `W_down(silu(W_gate x) ⊙ W_up x)` | Better quality per FLOP than GELU MLP | [Shazeer 2020](https://arxiv.org/abs/2002.05202) |
| GQA | Several query heads share one K/V head | KV cache ÷ (n_heads / n_kv_heads) at almost no quality loss | [Ainslie et al. 2023](https://arxiv.org/abs/2305.13245) |
| QK-norm | RMSNorm on q and k per head before RoPE | Bounds attention logits → fewer loss spikes | [Dehghani et al. 2023 (ViT-22B)](https://arxiv.org/abs/2302.05442), [OLMo 2](https://arxiv.org/abs/2501.00656) |
| Gated attention | `y = SDPA(q,k,v) ⊙ σ(W_g x)` (head-wise, elementwise) | Removes attention sinks and massive activations; fewer spikes; NeurIPS 2025 best paper; shipped in Qwen3-Next/3.5 | [Qiu et al. 2025](https://arxiv.org/abs/2505.06708), [code](https://github.com/qiuzh20/gated_attention) |
| No biases, tied/untied embeddings | Fewer params; tying helps small models | Small models tie (SmolLM, Gemma); large models untie | — |
| z-loss / logit soft-capping | Keeps output logits bounded | Late-training stability in bf16 | [PaLM](https://arxiv.org/abs/2204.02311), [Gemma 2](https://arxiv.org/abs/2408.00118) |

**In this repo:** every row above is implemented in [`aistudy/llm/model.py`](../aistudy/llm/model.py) (`RMSNorm`, `apply_rope`, `Attention` with `n_kv_heads`, `qk_norm`, `attn_gate`, `SwiGLU`, `MoE`, `z_loss`, `logit_softcap`).

**Recommended overview:** Sebastian Raschka, [*The Big LLM Architecture Comparison*](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison). It compares the architectures of the major open models side by side.

---

## 2. Attention: the main design space of 2025–26

The KV cache is the memory wall of inference:

```
KV bytes per token = 2 (K and V) × n_layers × n_kv_heads × head_dim × bytes_per_value
e.g. 64 layers × 8 KV heads × 128 dims × 2 bytes × 2 = 256 KiB/token  →  1M tokens ≈ 256 GiB per sequence
```

That is why every lab attacks attention.

| Family | Idea | Used in | Link |
|---|---|---|---|
| MHA → MQA → GQA | Share K/V heads across query heads | Llama 2/3, Qwen, Mistral, gpt-oss | [MQA (Shazeer 2019)](https://arxiv.org/abs/1911.02150), [GQA](https://arxiv.org/abs/2305.13245) |
| **MLA** (Multi-head Latent Attention) | Cache one low-rank latent vector per token; decompress K/V per head; decoupled RoPE | DeepSeek-V2/V3, Kimi K2 | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) |
| **Native Sparse Attention** | Compressed + selected + sliding-window branches, trainable end-to-end | Research (ACL 2025 best paper) | [Yuan et al. 2025](https://arxiv.org/abs/2502.11089) |
| **DeepSeek Sparse Attention (DSA)** | A small FP8 "lightning indexer" scores past tokens; attend only to top-k (k≈2048) | DeepSeek-V3.2 | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) |
| **CSA + HCA** | Compressed Sparse Attention (compress groups of KV entries, then sparse top-k) + Heavily Compressed Attention (dense attention over a highly compressed stream) | DeepSeek-V4 (Apr 2026): at 1M tokens V4-Pro reports 27% of V3.2's per-token FLOPs and 10% of its KV cache | [DeepSeek-V4 report](https://arxiv.org/abs/2606.19348) |
| Sliding-window / local–global interleave | Most layers see a local window; every k-th layer is global | Gemma 2/3, gpt-oss, Llama 4 (with NoPE global layers, "iRoPE") | [Gemma 3](https://arxiv.org/abs/2503.19786), [gpt-oss model card](https://arxiv.org/abs/2508.10925) |
| **Linear attention / SSM hybrids** | Replace most attention layers with recurrent layers whose state is fixed-size (O(1) memory per token) | Qwen3-Next & Qwen3.5 (3 Gated DeltaNet : 1 gated attention), Kimi Linear, NVIDIA Nemotron-H / Nemotron 3 | [Gated DeltaNet](https://arxiv.org/abs/2412.06464), [Kimi Linear](https://arxiv.org/abs/2510.26692), [Mamba-3 (ICLR 2026)](https://arxiv.org/abs/2603.15569) |
| Attention sinks | First tokens soak up attention mass; keep them in the cache window | Streaming inference; gpt-oss uses learned sink logits | [Xiao et al. 2023](https://arxiv.org/abs/2309.17453) |

**Conclusion:** for a new model under about 30B parameters, GQA with QK-norm and gating is the safe default. For long-context products, a 3:1 linear/full hybrid (Gated DeltaNet style) or an MLA-style compressed cache is the current frontier. Pure linear attention still underperforms on retrieval-heavy tasks, which is why everyone keeps some full-attention layers.

---

## 3. Mixture of Experts (MoE)

**Mechanics.** A router scores each token against E experts and sends it to the top-k. The output is the gate-weighted sum of those experts' FFN outputs, plus one or more **shared experts** that every token passes through. Compute scales with *k*, while capacity scales with *E*.

| Design choice | 2026 default | Notes |
|---|---|---|
| Granularity | Many small experts (64–384), k = 4–8 | "Fine-grained" experts specialise better ([DeepSeekMoE](https://arxiv.org/abs/2401.06066)) |
| Shared experts | 1 (sometimes 0) | Captures common knowledge |
| Router | Sigmoid affinities, normalised over the selected experts | DeepSeek-V3; softmax top-k also common |
| Load balancing | **Auxiliary-loss-free**: a per-expert bias steers selection only, updated by `b += γ·sign(mean_load − load)` | [Wang et al. 2024](https://arxiv.org/abs/2408.15664). A tiny sequence-level aux loss is often kept as a safety net |
| Router stability | Router z-loss, fp32 router | [ST-MoE](https://arxiv.org/abs/2202.08906) |
| Dense first layers | First 1–3 layers dense | Early layers do not benefit from routing |
| Systems | Expert parallelism + all-to-all, grouped GEMM, token dropping avoided | [MegaBlocks](https://arxiv.org/abs/2211.15841), DeepEP/DeepGEMM (DeepSeek open-source week) |

**Reported frontier configurations.** Check model cards before citing; secondary sources disagree on some numbers.

| Model (release) | Total / active params | Attention | Notes |
|---|---|---|---|
| DeepSeek-V3 (Dec 2024) | 671B / 37B | MLA | FP8 training, MTP, aux-loss-free MoE · [paper](https://arxiv.org/abs/2412.19437) |
| Kimi K2 (Jul 2025) | 1T / 32B | MLA | MuonClip optimizer, zero loss spikes over 15.5T tokens · [paper](https://arxiv.org/abs/2507.20534) |
| gpt-oss-120b (Aug 2025) | 117B / 5.1B | GQA, alternating banded-window/dense, attention sinks | [model card](https://arxiv.org/abs/2508.10925) |
| Qwen3-Next / Qwen3.5-397B-A17B (2025–26) | 397B / 17B (3.5) | 3× Gated DeltaNet : 1× gated attention | [Qwen3 report](https://arxiv.org/abs/2505.09388) (3.5 details from model cards / docs) |
| DeepSeek-V3.2 (Dec 2025) | 671B / 37B | MLA + DSA | [paper](https://arxiv.org/abs/2512.02556) |
| DeepSeek-V4-Pro / Flash (Apr 2026) | 1.6T / 49B and 284B / 13B | CSA + HCA hybrid, mHC residuals, Muon | 1M context, MIT license · [paper](https://arxiv.org/abs/2606.19348) |

Survey: [*The Evolution of MoE Architectures in LLMs*](https://arxiv.org/abs/2608.08650) (2026).

**In this repo:** `MoE` in [`model.py`](../aistudy/llm/model.py) implements sigmoid routing, top-k selection, a shared expert, aux-loss-free bias balancing (`update_bias`, all-reduced across ranks), an optional Switch-style aux loss, and active-parameter accounting. Train with `--n-experts 8 --moe-top-k 2` and watch `moe_max_load_ratio` move toward 1.0.

---

## 4. Residual stream, depth and multi-token prediction

- **Hyper-connections → mHC.** These widen the residual stream into *n* parallel streams mixed by learned matrices. DeepSeek's [mHC](https://arxiv.org/abs/2512.24880) constrains the mixing matrices to be doubly stochastic (Sinkhorn–Knopp projection), which restores the identity-mapping stability of plain residuals. DeepSeek-V4 uses it.
- **Multi-token prediction (MTP).** Extra heads or modules predict tokens t+2, t+3, … This gives a denser training signal and free draft tokens for speculative decoding ([Gloeckle et al. 2024](https://arxiv.org/abs/2404.19737); DeepSeek-V3 MTP module).
- **Depth vs width.** At fixed parameters, deeper-narrower models do better on reasoning in several 2025 ablations ([MobileLLM](https://arxiv.org/abs/2402.14905), the SmolLM3 playbook). Use µP/CompleteP to transfer hyper-parameters across depth (see [04-training-techniques.md](04-training-techniques.md)).

## 5. Positions and long context

RoPE with a large base (θ = 500k–10M) is the standard, with context extended in a late training phase through NTK-aware or [YaRN](https://arxiv.org/abs/2309.00071) scaling. Several 2025 models apply RoPE to only part of the head dimension (partial RoPE), or interleave NoPE (no positional encoding) global layers (Llama 4 "iRoPE", SmolLM3).

Million-token contexts are now shipped (DeepSeek-V4, Qwen hosted models). The bottleneck is the KV cache, hence section 2. Evaluate long context with retrieval and reasoning tasks (RULER, needle variants), not perplexity.

## 6. Tokenization

- **The standard** is byte-level BPE with 100k–260k vocabularies: Llama 3 uses 128k, Gemma 3 uses 262k. Larger vocabularies compress better, especially for multilingual text and code.
- **Tokenizer-free alternatives** are maturing: [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) uses dynamic byte patches.
- **In this repo** we use a byte tokenizer so that everything runs without downloads. That is also why the reported "val_bpb" (bits per byte) is directly comparable across tokenizers.

## 7. Beyond the autoregressive Transformer

| Direction | Status (Nov 2026) | Links |
|---|---|---|
| State-space / linear RNNs | Pure SSMs lag on recall; they win as **hybrid layers**. Mamba-3 (ICLR 2026) adds complex-valued state and MIMO updates | [Mamba](https://arxiv.org/abs/2312.00752), [Mamba-2](https://arxiv.org/abs/2405.21060), [Mamba-3](https://arxiv.org/abs/2603.15569) |
| Diffusion LMs | Commercial (Inception Mercury 2, Gemini Diffusion) and open (LLaDA 8B → LLaDA 2.x). Their advantage is parallel decoding (5–10× faster). *Block diffusion* (autoregressive across blocks, diffusion within) keeps the KV cache | [LLaDA](https://arxiv.org/abs/2502.09992), [MDLM](https://arxiv.org/abs/2406.07524), [Block Diffusion](https://arxiv.org/abs/2503.09573), [survey](https://arxiv.org/abs/2508.10875), [Kuleshov group tutorial (2026)](https://kuleshov-group.github.io/blog/blog/2026/how-to-build-a-diffusion-language-model/) |
| Looped / recurrent-depth Transformers | Reuse layers to "think longer" in latent space; research stage | [Geiping et al. 2025](https://arxiv.org/abs/2502.05171) |

## 8. Scaling laws: how big, how long

- [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) showed that loss follows power laws in parameters, data and compute.
- [Chinchilla (Hoffmann et al. 2022)](https://arxiv.org/abs/2203.15556) found that compute-optimal training uses about **20 tokens per parameter**.
- Modern practice is to **over-train** small models far past Chinchilla (Llama 3 8B saw 15T tokens, roughly 1,900 tokens/param), because inference cost dominates over a model's lifetime.
- When data runs out, repeat it: up to about 4 epochs is nearly as good as fresh data ([Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264)).
- Practical rule: fit your own scaling curve on 5–8 small runs before committing to a large one. Epoch AI tracks the empirical trends at [epoch.ai](https://epoch.ai).

## 9. Reading list, in order

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
2. [Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783): the most complete "dense recipe" paper
3. [OLMo 2](https://arxiv.org/abs/2501.00656): fully open data, code and logs, with stability ablations
4. [DeepSeek-V3](https://arxiv.org/abs/2412.19437) → [V3.2](https://arxiv.org/abs/2512.02556) → [V4](https://arxiv.org/abs/2606.19348): the MoE / FP8 / sparse-attention frontier
5. [Kimi K2](https://arxiv.org/abs/2507.20534): Muon at trillion scale and agentic post-training
6. [Qwen3 technical report](https://arxiv.org/abs/2505.09388) and [Gated Attention](https://arxiv.org/abs/2505.06708)
7. [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) (Hugging Face, Oct 2025): every decision behind SmolLM3, including the failures
8. [Physics of Language Models](https://physics.allen-zhu.com/): controlled experiments on what Transformers can learn

## 10. Exercises with this repo

1. Train `--preset tiny` with and without `--no-qk-norm --no-attn-gate` at a high LR (`--lr 1e-2`). Compare `grad_norm` traces in `runs/llm/log.jsonl`.
2. Train a dense and an MoE model with the same *active* parameter count. Compare val loss at equal tokens.
3. `python -m aistudy.llm.generate --ckpt runs/llm/best.pt --benchmark` measures the KV-cache speed-up. Explain the asymptotics.
4. Train a 2-layer "draft" and a 6-layer "target", then run speculative decoding (`--draft-ckpt`). Relate the acceptance rate to the draft/target loss gap.
5. Read `KVCache` and `_attention_mask`, then extend the cache with a sliding window (local attention) and test it against the full forward pass.
