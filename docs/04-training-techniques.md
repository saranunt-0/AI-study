# 04 · Training Techniques: the Full Recipe

> **Code:** [`aistudy/common/optim.py`](../aistudy/common/optim.py) (AdamW groups, Muon, schedules) · [`aistudy/common/utils.py`](../aistudy/common/utils.py) (seeding, autocast, DDP init, atomic checkpoints) · [`aistudy/common/ema.py`](../aistudy/common/ema.py) · the three `train.py` scripts

## Short conclusion

- **Data is the model.**
  - Teams that win spend most of their effort on data: sourcing, filtering, deduplication, mixing and annealing, plus evaluation.
  - Architecture and optimizer choices are usually worth a few percent.
  - Data quality is often worth 2–10× of effective compute ([FineWeb](https://arxiv.org/abs/2406.17557), [DCLM](https://arxiv.org/abs/2406.11794)).
- **Defaults that just work in 2026:**
  - **Optimizer:** AdamW (β = 0.9/0.95, weight decay 0.1 on matrices only), or **Muon** on hidden matrices plus AdamW for the rest. Muon is now proven at trillion scale (Kimi K2) and used in DeepSeek-V4.
  - **Schedule:** warmup, then cosine or **WSD**.
  - **Precision and clipping:** bf16 autocast with fp32 master weights; gradient clipping at 1.0.
  - **Stability:** QK-norm and z-loss.
  - **Hyper-parameters:** µP-style parameterisation, so they transfer from small proxies.
- **Engineering multiplies everything.**
  - Hit 40–60% MFU (FSDP2/ZeRO, tensor/pipeline/expert/context parallelism, FlashAttention, torch.compile).
  - Save atomic checkpoints often.
  - Make every run reproducible and resumable.
- **Scientific method beats intuition.** Run ablations at small scale with fixed seeds and multiple evals, fit scaling curves, then scale. Read the [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) for an honest account of how a real team does this.

---

## 1. Data

| Stage | Practice | References |
|---|---|---|
| Sourcing | Web (Common Crawl), code, math, books, papers, synthetic | [FineWeb / FineWeb-Edu](https://arxiv.org/abs/2406.17557), [DCLM](https://arxiv.org/abs/2406.11794), [The Stack v2](https://arxiv.org/abs/2402.19173) |
| Filtering | Heuristics → model-based quality classifiers (e.g. "educational value") | FineWeb-Edu classifier, DCLM fastText filter |
| Deduplication | Exact + fuzzy (MinHash-LSH) at document and paragraph level | [Lee et al. 2021](https://arxiv.org/abs/2107.06499) |
| Decontamination | Remove benchmark overlap (n-gram), or results are meaningless | — |
| Mixing | Tune domain weights with small proxy models | [DoReMi](https://arxiv.org/abs/2305.10429), [RegMix](https://arxiv.org/abs/2407.01492) |
| Annealing / mid-training | Up-weight the highest-quality data (math, code, reasoning, long context) during the LR decay phase | Llama 3, OLMo 2, SmolLM3 |
| Synthetic data | Rephrasing, textbook-style generation, verified reasoning traces | [Phi-1 "Textbooks are all you need"](https://arxiv.org/abs/2306.11644), Nemotron-CC |
| Packing | Concatenate documents with EOS; optionally mask cross-document attention | This repo: `aistudy/llm/data.py` |

## 2. Parameterisation and initialisation

- **Standard init:**
  - N(0, 0.02) for linear and embedding layers;
  - residual output projections scaled by `1/√(2L)` (GPT-2/Megatron), implemented in `Transformer.__init__`;
  - zero-init for gates and adaLN (the DiT case).
- **µP (Maximal Update Parametrization).** Scales init, LR and output multipliers with width so that **the optimal LR found on a small model transfers to a large one** ([Tensor Programs V](https://arxiv.org/abs/2203.03466); [practitioner guide by EleutherAI/Cerebras](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization)). **CompleteP** extends transfer across *depth*: α = 1 for residual-branch scaling, with a reported 12–34% compute savings ([Dey et al., NeurIPS 2025](https://arxiv.org/abs/2505.01618)).
- **Sanity check at step 0.** The initial loss must be ≈ ln(vocab_size). For this repo's byte model that is ln(264) ≈ 5.58; the logs show 5.6.

## 3. Optimizers

| Optimizer | When | Key settings | Link |
|---|---|---|---|
| **AdamW** | Default for everything | β₁ = 0.9, β₂ = 0.95 (LLMs) / 0.99–0.999 (vision, diffusion), ε = 1e-8, wd = 0.1 on ndim ≥ 2 only | [Loshchilov & Hutter](https://arxiv.org/abs/1711.05101) |
| **Muon** | Hidden 2-D weights of Transformers; ~1.3–2× token-efficiency over AdamW in speedruns and at scale | Nesterov momentum 0.95 → Newton–Schulz orthogonalisation (5 steps) → scale to match AdamW RMS (0.2·√max(m,n)) → decoupled wd | [Keller Jordan blog](https://kellerjordan.github.io/posts/muon/), [Moonlight: Muon is Scalable](https://arxiv.org/abs/2502.16982), [Bernstein: Deriving Muon](https://jeremybernste.in/writing/deriving-muon) |
| **MuonClip** | Muon at trillion scale | Muon + **QK-Clip**: rescale W_q, W_k when the max attention logit exceeds τ (=100). Kimi K2 trained 15.5T tokens with no loss spikes | [Kimi K2](https://arxiv.org/abs/2507.20534) |
| Shampoo / SOAP | Second-order-ish preconditioning; won the AlgoPerf benchmark | More memory, more engineering | [Shampoo](https://arxiv.org/abs/1802.09568), [SOAP](https://arxiv.org/abs/2409.11321) |
| Lion | Sign-based, memory-light | Use ~3–10× smaller LR than AdamW | [arXiv:2302.06675](https://arxiv.org/abs/2302.06675) |
| Schedule-free AdamW | No schedule needed; any-time checkpoints | Good for unknown training lengths | [arXiv:2405.15682](https://arxiv.org/abs/2405.15682) |
| Adafactor / 8-bit Adam | Memory-constrained fine-tuning | Factored or quantised optimizer state | [Adafactor](https://arxiv.org/abs/1804.04235), [8-bit](https://arxiv.org/abs/2110.02861) |

**Why Muon works (one paragraph).** Gradient updates for weight matrices are typically very low-rank: a few singular directions dominate. Orthogonalising the momentum, `G = USVᵀ → UVᵀ`, gives every direction the same step size. It is steepest descent under the spectral norm, which is the natural norm for a linear layer acting on hidden states. Embeddings, LM heads, norms and biases are not "matrices acting on hidden states", so they stay on AdamW. `build_optimizer(kind="muon")` in this repo implements exactly that split.

**Benchmark your optimizer ideas on** [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), the GPT-2 speedrun where Muon was born, and on [AlgoPerf](https://github.com/mlcommons/algorithmic-efficiency).

## 4. Learning-rate schedules and batch size

- **Warmup** (1–5% of steps, or ~2k steps for LLMs) prevents early divergence while Adam's second-moment estimate is noisy.
- **Cosine** decays to about 10% of peak. It requires knowing the total number of steps.
- **WSD (warmup–stable–decay)** holds the LR constant, then decays linearly or as 1−√ in the last 10–20% of steps. You can branch a decay from any checkpoint, which makes it ideal for continual pre-training and data ablations ([MiniCPM](https://arxiv.org/abs/2404.06395), [Hägele et al.](https://arxiv.org/abs/2405.18392)). Used by DeepSeek, Kimi and SmolLM3. Implemented in `lr_schedule(kind="wsd")`.
- **Batch size and the critical batch size.**
  - Below the *gradient noise scale* B_noise, doubling the batch nearly halves the steps; above it you waste compute ([McCandlish et al. 2018](https://arxiv.org/abs/1812.06162)).
  - B_noise grows as the loss falls, which is why batch-size warmup is common.
  - This repo estimates it online: `python -m aistudy.llm.train --grad-accum 4 --noise-scale` logs `noise_scale_tokens`. In our MoE run it grew from ~360 to ~7,300 tokens as the loss fell from 2.9 to 1.4.
- **LR vs batch size.** For Adam, the optimal LR grows roughly with √batch until the critical batch size; re-tune when you change the batch size by more than ~4×.

## 5. Numerics and precision

| Format | Use | Notes |
|---|---|---|
| fp32 | Master weights, optimizer state, reductions, softmax/norm statistics | Never drop this for the master copy |
| **bf16** | Default compute dtype (autocast) | Same exponent range as fp32 → no loss scaling |
| fp16 | Older GPUs | Requires `GradScaler` (dynamic loss scaling); this repo supports `--dtype fp16` |
| **FP8** | Matmuls at scale (Hopper/Blackwell) | Fine-grained (tile/block-wise) scaling as in DeepSeek-V3; keep embeddings, heads, norms and attention softmax in higher precision |
| **NVFP4** | 4-bit pre-training on Blackwell | 16-element blocks with E4M3 scales, random Hadamard transforms, stochastic rounding, 2-D scaling; 12B model on 10T tokens matched FP8 ([NVIDIA 2025](https://arxiv.org/abs/2509.25149)) |

## 6. Distributed training

| Technique | Shards | Use when | Reference |
|---|---|---|---|
| DDP | Nothing (replicates model) | Model + optimizer fit on one GPU | This repo (`torchrun ... -m aistudy.llm.train`) |
| ZeRO-1/2/3 / **FSDP2** | Optimizer state / grads / params | Model doesn't fit; the default first step | [ZeRO](https://arxiv.org/abs/1910.02054), [PyTorch FSDP](https://arxiv.org/abs/2304.11277) |
| Tensor parallel (TP) | Individual matmuls | Within a node (NVLink) | [Megatron-LM](https://arxiv.org/abs/1909.08053) |
| Pipeline parallel (PP) | Layers | Across nodes; needs micro-batching (1F1B, zero-bubble) | [GPipe](https://arxiv.org/abs/1811.06965), [Zero Bubble PP](https://arxiv.org/abs/2401.10241) |
| Context / sequence parallel | The sequence | Long context (ring attention) | [Ring Attention](https://arxiv.org/abs/2310.01889) |
| Expert parallel (EP) | MoE experts | MoE models; all-to-all communication | DeepSeek-V3 report |

**The two best free books on this:**
- [**The Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (Hugging Face), which covers every parallelism with measurements;
- [**How to Scale Your Model**](https://jax-ml.github.io/scaling-book/) (Google DeepMind), on rooflines, TPUs and JAX sharding.

**Frameworks:** [torchtitan](https://github.com/pytorch/torchtitan) (PyTorch-native reference), [Megatron-LM/Core](https://github.com/NVIDIA/Megatron-LM), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), [MaxText](https://github.com/AI-Hypercomputer/maxtext), [nanotron](https://github.com/huggingface/nanotron).

## 7. Throughput: making every FLOP count

- **MFU** (model FLOPs utilisation) = achieved FLOPs / peak FLOPs. Training FLOPs per token ≈ `6·N_active + 12·L·H·d_head·T` ([PaLM App. B](https://arxiv.org/abs/2204.02311)). This repo logs it with `--peak-tflops`. Good runs reach 40–60%.
- **Kernels.** FlashAttention [1](https://arxiv.org/abs/2205.14135) / [2](https://arxiv.org/abs/2307.08691) / [3](https://arxiv.org/abs/2407.08608), fused RMSNorm/SwiGLU/cross-entropy ([Liger kernels](https://github.com/linkedin/Liger-Kernel)), `torch.compile` (`--compile`), and Triton for custom kernels ([GPU MODE lectures](https://github.com/gpu-mode/lectures)).
- **Memory.**
  - Activation checkpointing (`--grad-checkpointing`) trades ~33% more compute for O(L) less activation memory.
  - Use sequence packing and avoid padding.
  - Keep the vocabulary a multiple of 64 or 128.
- **Input pipeline.** Pinned memory + non-blocking copies (in `TokenStream.get_batch`); pre-tokenised memory-mapped shards (`python -m aistudy.llm.data`).

## 8. Experiment methodology (what separates world-class teams)

1. **Evaluation first.** Decide the metrics *before* training: val loss/bpb, a small set of downstream evals with low noise at small scale, and a held-out decontaminated set.
2. **Ablate at small scale, one change at a time, with ≥2 seeds.** Measure noise: if seed-to-seed variance exceeds your effect size, the ablation is meaningless.
3. **Fit scaling curves** (loss vs compute) for your top-2 candidate recipes; pick the one with the better slope, not the better point.
4. **Keep a "golden" baseline config** in version control and re-run it after every infrastructure change.
5. **Log everything** (see [05-gradient-control-and-stability.md](05-gradient-control-and-stability.md) for *what*): loss, grad norm, LR, throughput, per-layer stats, data mix.
6. **Checkpoint atomically and often** (`save_checkpoint` writes to a tmp file then renames). Test resume as part of CI; this repo's resume path was tested to continue exactly at step 150.
7. **Write a run log** of decisions, anomalies and fixes. The Smol Training Playbook and the [OLMo](https://arxiv.org/abs/2501.00656) / [Llama 3](https://arxiv.org/abs/2407.21783) reports are excellent models of this.

## 9. Recipes by model type (cheat sheet)

| Knob | LLM pre-training | Diffusion (DiT) | World model (latent) |
|---|---|---|---|
| Optimizer | AdamW (0.9, 0.95) or Muon | AdamW (0.9, 0.99–0.999), wd 0–0.01 | AdamW (0.9, 0.99) |
| Peak LR (small models) | 1e-3–3e-3 | 1e-4–5e-4 | 3e-4–1e-3 |
| Schedule | Warmup + cosine / WSD | Warmup + constant (EMA does the averaging) | Warmup + cosine |
| Clipping | 1.0 | 1.0 | 1.0 |
| Special | z-loss, QK-norm, MoE balancing | EMA 0.999–0.9999, label dropout 10%, logit-normal t | Anti-collapse term (SIGReg/VICReg/EMA), multi-step rollout loss |

## 10. Learn more

- Course: [Stanford CS336 *Language Modeling from Scratch*](https://cs336.stanford.edu/), which covers tokenizer → kernels → parallelism → scaling laws → data → RL.
- Repos: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), [karpathy/nanochat](https://github.com/karpathy/nanochat) (a full ChatGPT-style pipeline for about $100 on 8×H100), [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
- Books: [Stas Bekman, *Machine Learning Engineering Open Book*](https://github.com/stas00/ml-engineering), covering hardware, debugging and fault tolerance.
