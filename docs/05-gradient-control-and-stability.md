# 05 · Gradient Control and Training Stability

> **Code:** [`aistudy/common/grad.py`](../aistudy/common/grad.py): `clip_grad_norm_`, `adaptive_grad_clip_`, `SpikeGuard`, `GradNoiseScale`, `per_parameter_grad_norms`, `update_to_weight_ratio`, `z_loss`; wired into [`aistudy/llm/train.py`](../aistudy/llm/train.py) with `--grad-clip`, `--agc`, `--spike-guard`, `--noise-scale`

## Short conclusion

- **Divergence and loss spikes are the most expensive failures in large-scale training.** A spike at 70% of a $10M run costs weeks. Most spikes come from a handful of well-understood mechanisms:
  - attention-logit growth;
  - output-logit drift;
  - a stale Adam second moment for rarely-updated parameters (embeddings);
  - LR too high for the batch size;
  - bad data batches.
- **Architectural prevention beats runtime cures.** Use pre-norm + RMSNorm, **QK-norm** (or **QK-Clip** with Muon), z-loss / logit soft-capping, gated attention, scaled residual init, warmup, and β₂ = 0.95.
- **Runtime control is the safety net:**
  - clip the global gradient norm at 1.0;
  - skip outlier steps automatically;
  - when spikes persist, **rewind to an earlier checkpoint and skip the offending data** (PaLM).
- **Monitor the right signals:** loss, global grad-norm (pre-clip), per-layer grad-norm, update/weight ratio, max attention logit, activation RMS, and MoE expert load. NaNs are the *last* symptom; the grad norm usually drifts upward hundreds of steps earlier.

---

## 1. Why gradients explode or vanish

For a deep network, gradients are products of layer Jacobians. If their singular values are consistently above or below 1, gradients explode or vanish exponentially with depth ([Bengio et al. 1994](https://ieeexplore.ieee.org/document/279181), [Pascanu et al. 2013](https://arxiv.org/abs/1211.5063)). Modern architectures keep signal propagation near-isometric:

| Mechanism | How it helps | In this repo |
|---|---|---|
| Residual connections | Identity path: ∂x_L/∂x_l contains I | Every block |
| Pre-norm | Residual stream never normalised → clean identity path; sub-layer inputs well-scaled | `Block` (LLM), `DiTBlock` |
| Scaled residual init (1/√2L) | Residual-stream variance doesn't grow with depth at init | `Transformer.__init__` |
| Zero-init of last layers / gates | Each block starts as identity (adaLN-Zero, ReZero, Fixup) | `DiT._init_weights`, world-model predictor `out` |
| Normalisation of q, k | Attention logits bounded regardless of weight growth | `qk_norm=True` |
| Doubly-stochastic residual mixing (mHC) | Hyper-connections without losing identity-mapping stability | (see [mHC](https://arxiv.org/abs/2512.24880)) |

## 2. Known causes of loss spikes and their fixes

| Cause | Symptom | Fix | Evidence |
|---|---|---|---|
| **Attention logit growth** (‖q‖·‖k‖ grows → softmax saturates → one-hot attention) | Max attention logit climbs; spikes worse at high LR | QK-norm; **QK-Clip** (rescale W_q, W_k when max logit > τ) | [Dehghani et al. 2023](https://arxiv.org/abs/2302.05442), [Wortsman et al. 2023](https://arxiv.org/abs/2309.14322), [Kimi K2 MuonClip](https://arxiv.org/abs/2507.20534) |
| **Output logit divergence** (log Z drifts) | Loss plateaus then jumps; bf16 round-off in softmax | z-loss 1e-4·log²Z; logit soft-cap (30) | [PaLM](https://arxiv.org/abs/2204.02311), [Gemma 2](https://arxiv.org/abs/2408.00118) |
| **Adam ε / β₂ mismatch**: a rarely-seen token's embedding gets a huge update when v̂ is tiny and stale | Spikes correlated with rare tokens / new data | β₂ = 0.95; smaller ε (1e-8 → 1e-15 at large scale); embedding LR scaling | [Wortsman et al.](https://arxiv.org/abs/2309.14322), [Molybog et al. "A Theory on Adam Instability"](https://arxiv.org/abs/2304.09871) |
| **Massive activations / attention sinks** | A few dimensions reach 1000× the median | Gated attention; learned sinks | [Sun et al. 2024](https://arxiv.org/abs/2402.17762), [Gated Attention](https://arxiv.org/abs/2505.06708) |
| **LR too high for batch size / no warmup** | Divergence in the first 1k steps | Longer warmup; LR sweep at small scale + µP transfer | — |
| **Bad data** (corrupted, extremely repetitive or adversarial batches) | Isolated spikes that reproduce when replaying the same batch | Skip the batch; fix the filter; rewind + skip | PaLM §5.1, OLMo reports |
| **MoE router collapse** | Few experts take all tokens; dead experts | Aux-loss-free bias balancing; router z-loss; fp32 router | [ST-MoE](https://arxiv.org/abs/2202.08906), [Wang et al. 2024](https://arxiv.org/abs/2408.15664) |
| **Precision** | NaN/Inf in fp16; slow drift in bf16 | bf16 + fp32 master weights; fp16 needs dynamic loss scaling; FP8/FP4 need fine-grained scaling | [Micikevicius et al.](https://arxiv.org/abs/1710.03740), [NVFP4](https://arxiv.org/abs/2509.25149) |

**Spike recipe (production).**
1. Automatic skip of non-finite or outlier steps (`SpikeGuard`).
2. If the guard fires repeatedly, stop and rewind to the checkpoint ~100–500 steps earlier.
3. Skip the 200–500 batches around the spike and lower the LR temporarily.
4. Investigate offline by replaying those batches on the saved checkpoint.

## 3. Runtime gradient control: the tools

| Tool | Formula | Default | Use |
|---|---|---|---|
| **Global-norm clipping** | g ← g · min(1, c/‖g‖) | c = 1.0 | Every LLM/diffusion run. Log the *pre-clip* norm. If clipping fires on >5% of steps, your LR is too high |
| Value clipping | g ← clip(g, −c, c) | — | Rarely; distorts direction |
| **AGC** (adaptive, unit-wise) | g_i ← g_i · min(1, λ·‖w_i‖/‖g_i‖) | λ = 0.01 | Normaliser-free nets, very large batches ([NFNets](https://arxiv.org/abs/2102.06171)); `--agc 0.01` |
| Adaptive / z-score clipping | Clip relative to running statistics of ‖g‖ | — | Recent proposals (e.g. ZClip, AdaGC); `SpikeGuard` implements the "skip" variant |
| **Skip-step** | Don't apply the update if loss or ‖g‖ is a >6σ outlier or non-finite | warmup 50, max 5 in a row | `--spike-guard`; identical decision on every rank (loss is all-reduced first) |
| Gradient accumulation | Sum micro-batch grads, step once | — | Large effective batch on small hardware; use DDP `no_sync()` on all but the last micro-batch |
| Gradient noise scale | B_noise = tr(Σ)/‖G‖² | — | Pick the batch size; `--noise-scale --grad-accum 4` |
| Loss scaling (fp16) | Multiply loss by S, unscale grads before clipping | dynamic | `torch.amp.GradScaler`; **unscale before clipping** (done in `train.py`) |

**Order of operations per optimizer step.** This is a common bug source; `aistudy/llm/train.py` follows it exactly:

```
for micro in micro_batches:  loss/accum → backward   (no_sync except last micro-batch)
all-reduce loss for logging/decisions (DDP)
scaler.unscale_(opt)                  # fp16 only
grad_norm = global_norm(grads)        # log PRE-clip
[optional] AGC
clip_grad_norm_(params, 1.0)
if SpikeGuard says skip: zero grads, scaler.update(), continue
optimizer.step(); scaler.update(); zero_grad(set_to_none=True)
MoE bias update / EMA update
```

## 4. What to monitor (dashboard checklist)

| Signal | Healthy | Warning sign |
|---|---|---|
| Train loss (smoothed) & val loss | Smooth, monotone decrease; val tracks train | Spikes; val diverges from train (overfitting or leakage) |
| Global grad-norm (pre-clip) | Stable or slowly decreasing after warmup | Upward drift, or clipping on most steps |
| Per-layer grad-norm (`per_parameter_grad_norms`) | Similar order of magnitude across layers | One layer (often embeddings or the last block) dominates |
| Update/weight ratio (`update_to_weight_ratio`) | ~1e-3 per step (AdamW) | >1e-2: LR too high for that tensor |
| Max attention logit | Bounded (QK-norm) | Monotone growth |
| Activation RMS per layer | Flat across depth | Growth with depth → massive activations |
| MoE expert load (max/mean) | → ~1–1.5 | Persistent >3, dead experts |
| Throughput / MFU | Constant | Drops → stragglers, data loader, thermal throttling |
| Initial loss | ≈ ln(V) | Much larger → init or label bug |

## 5. Stability in RL post-training (2025–26)

RL with verifiable rewards has its own failure modes ([survey](https://arxiv.org/abs/2407.16216)):
- **Entropy collapse.** The policy becomes deterministic too early. DAPO uses "clip-higher", asymmetric clipping of the importance ratio ([DAPO](https://arxiv.org/abs/2503.14476)).
- **Length bias from per-sequence normalisation.** Fixed by Dr. GRPO ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)).
- **Token-level importance ratios are noisy for long sequences.** GSPO uses sequence-level ratios, which stabilises MoE RL ([arXiv:2507.18071](https://arxiv.org/abs/2507.18071)).
- **Training–inference mismatch.** Logprobs from the inference engine (vLLM/SGLang) differ from the trainer's, so use importance-sampling corrections. This is a hot topic in 2025–26 large-scale RL.
- **Reward hacking.** The policy exploits the grader. Treat it as a *stability* problem *and* an alignment problem (see [09-ai-philosophy-and-ethics.md](09-ai-philosophy-and-ethics.md)).

## 6. Debugging playbook (in order)

1. **Overfit one batch.** The loss must go to ~0 (`test_llm_overfits_one_batch`). If not, you have a bug, not a tuning problem.
2. **Check the initial loss** equals ln(V), and that the loss with an all-zero input is not lower than with real input.
3. **Run in fp32 with deterministic algorithms** (`set_seed(..., deterministic=True)`). If the problem disappears, it is numerical.
4. **Halve the LR.** If the spike moves later but persists, it is architectural (logit growth); if it vanishes, the LR is too high.
5. **Replay the exact batch** at the spiking step from the checkpoint before it. Data or optimizer state?
6. **Bisect configs** against the golden baseline, one change at a time.
7. **Check distributed invariants.** Same weights on all ranks after a step, identical skip decisions, and identical MoE biases (this repo all-reduces expert loads).

## 7. Further reading

- [Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities*](https://arxiv.org/abs/2309.14322): reproduce big-model instabilities on small models. The single best paper on this topic.
- [OLMo 2](https://arxiv.org/abs/2501.00656): stability ablations with full logs (QK-norm, reordered norm, z-loss, init).
- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook): "the training marathon" and cluster debugging.
- [Karpathy, *A Recipe for Training Neural Networks*](https://karpathy.github.io/2019/04/25/recipe/): the classic debugging mindset.
- [Stas Bekman, ML Engineering: debugging and instabilities](https://github.com/stas00/ml-engineering).
