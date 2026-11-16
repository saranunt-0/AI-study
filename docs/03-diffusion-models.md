# 03 · Diffusion and Flow Models

> **Code:** [`aistudy/diffusion/`](../aistudy/diffusion): `model.py` (DiT with adaLN-Zero) · `flow.py` (flow-matching loss, Euler/Heun samplers, classifier-free guidance, v- and x-prediction) · `train.py` · `sample.py` (inference, trajectory plots, guidance sweep with a classifier-based metric)

## Short conclusion

- **The 2026 production recipe** is:
  - a **Diffusion Transformer** (DiT / MM-DiT);
  - trained with **flow matching / rectified flow**, with logit-normal timestep sampling;
  - in a **latent space** (a VAE, or increasingly a frozen representation encoder: RAE);
  - with **classifier-free guidance** at sampling time;
  - then **distilled to 1–8 steps** for deployment.
  
  This recipe powers SD3, FLUX, Movie Gen, Veo/Sora-class video and NVIDIA Cosmos.
- **DDPM, score SDEs, EDM and flow matching are one family.** They differ in the noise path, what the network predicts (ε, x₀, v) and the loss weighting. Learn the unified view once (Lai, Song et al. 2025 monograph; MIT 6.S184) and every paper becomes a small delta.
- **Frontier trends (2025–26):**
  1. **Representation Autoencoders.** Frozen DINO/SigLIP/MAE encoders with trained decoders as the latent space. They converge faster and need no REPA.
  2. **Pixel-space DiTs.** JiT, "Back to Basics" (Li & He, CVPR 2026): predict clean `x₀`, not noise, and use large patches with no tokenizer.
  3. **One- and few-step generation.** Consistency / sCM, shortcut models, MeanFlow.
  4. **Diffusion for text.** LLaDA, Mercury, block diffusion.
- **Practical takeaway.** Most quality comes from the data, the autoencoder/latent space, and guidance or sampler tuning, not from exotic losses. Always sample from EMA weights and measure with a fixed protocol (FID / FD-DINOv2 + human evaluation).

---

## 1. The unified picture in five equations

```
forward (noising) path      x_t = α_t · x_0 + σ_t · ε                 ε ~ N(0, I)
    DDPM / VP:              α_t² + σ_t² = 1
    flow matching / RF:     α_t = 1 − t,  σ_t = t                     (straight line, t ∈ [0,1])
what the net predicts       ε (DDPM) · x_0 (JiT) · v = dx_t/dt (RF) · score ∇log p_t = −ε/σ_t
loss                        E_t w(t) ‖ prediction − target ‖²          (w(t) and p(t) = the "design space")
sampling                    integrate the probability-flow ODE (or reverse SDE) from noise to data
guidance                    v̂ = v_uncond + w · (v_cond − v_uncond)     (classifier-free guidance)
```

| Milestone | Contribution | Link |
|---|---|---|
| DDPM (Ho et al. 2020) | Simple ε-prediction loss; diffusion becomes competitive | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) |
| DDIM | Deterministic, few-step sampling | [arXiv:2010.02502](https://arxiv.org/abs/2010.02502) |
| Score SDE (Song et al. 2021) | Continuous-time view; probability-flow ODE | [arXiv:2011.13456](https://arxiv.org/abs/2011.13456), [Yang Song's blog](https://yang-song.net/blog/2021/score/) |
| Classifier-free guidance | One network for conditional + unconditional; trade diversity for fidelity | [arXiv:2207.12598](https://arxiv.org/abs/2207.12598) |
| EDM (Karras et al. 2022) | Clean separation of design choices: preconditioning, σ-schedule, Heun sampler | [arXiv:2206.00364](https://arxiv.org/abs/2206.00364) |
| Latent Diffusion / Stable Diffusion | Diffuse in a VAE latent → 10–50× cheaper | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) |
| v-prediction | Better-conditioned target at high noise | [Salimans & Ho 2022](https://arxiv.org/abs/2202.00512) |
| **Flow matching / rectified flow** | Regress the velocity of a straight path; simple and fast | [Lipman et al.](https://arxiv.org/abs/2210.02747), [Liu et al.](https://arxiv.org/abs/2209.03003), [stochastic interpolants](https://arxiv.org/abs/2303.08797) |
| **DiT** | Transformer backbone with adaLN-Zero conditioning; scales like LLMs | [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) |
| SiT | DiT + interpolant framework ablations | [arXiv:2401.08740](https://arxiv.org/abs/2401.08740) |
| **SD3 (MM-DiT)** | Rectified flow at scale; logit-normal t; joint text-image attention; QK-norm | [arXiv:2403.03206](https://arxiv.org/abs/2403.03206) |
| Movie Gen | Flow matching + Transformer for video at 30B scale | [arXiv:2410.13720](https://arxiv.org/abs/2410.13720) |

**Best single references:**
- [*The Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890) (Lai, Song, Kim, Mitsufuji, Ermon, 2025), a full monograph;
- [*Flow Matching Guide and Code*](https://arxiv.org/abs/2412.06264) (Meta, 2024) with the [`flow_matching` library](https://github.com/facebookresearch/flow_matching);
- [MIT 6.S184: Generative AI with SDEs](https://diffusion.csail.mit.edu/), lecture notes and labs;
- [Sander Dieleman's blog](https://sander.ai/), for intuitions on guidance, noise schedules and "diffusion is spectral autoregression".

## 2. Architecture

| Choice | 2026 default | Notes |
|---|---|---|
| Backbone | DiT / MM-DiT (U-Nets mostly retired) | Text tokens and image tokens attend jointly (SD3, FLUX) or via cross-attention |
| Conditioning | adaLN-Zero (shift/scale/gate regressed from t + class/pooled text) | Zero-init gates → each block starts as identity |
| Positions | 2-D/3-D RoPE (variable resolution, video) | Fixed sincos in the original DiT (and this repo) |
| Stability | QK-norm, RMSNorm, bf16 | Same tricks as LLMs |
| Patch size | 2 in latent space; 16–32 in pixel space (JiT) | Tokens ∝ (H·W)/p² → compute ∝ p⁻⁴ |

## 3. Latent space: the hidden lever

| Approach | Idea | Link |
|---|---|---|
| VAE (SD-VAE, FLUX-VAE) | KL-regularised conv autoencoder, 8× downsampling, 4–16 channels | [LDM](https://arxiv.org/abs/2112.10752) |
| REPA | Align DiT hidden states with DINOv2 features → much faster convergence | [arXiv:2410.06940](https://arxiv.org/abs/2410.06940) |
| **RAE** (ICLR 2026) | Use a *frozen* representation encoder (DINOv2, SigLIP2, MAE) with a trained decoder as the autoencoder; a wide shallow "DDT head" handles the high-dimensional latents | [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690) |
| Pixel space again | No tokenizer; large patches; x-prediction | [JiT / Back to Basics (Li & He)](https://arxiv.org/abs/2511.13720) |

**Why x-prediction matters (JiT).** Natural images lie on a low-dimensional manifold, while noise does not. With 16×16×3 = 768-dimensional patches, a network predicting ε or v must output an off-manifold, full-rank target. Predicting clean `x₀` is much easier. The JiT ablation on ImageNet 256 with JiT-B/16 gives FID 8.6 for x-prediction, versus 96.5 for v-prediction and 372 for ε-prediction.

**In this repo:** `--prediction x` implements exactly this conversion (`v = (x_t − x̂₀)/t`, with t clamped). Small 32×32 images with 4×4 patches won't show a large gap. Try `--patch-size 8` or `16` to see it.

## 4. Fast sampling and distillation

| Method | Steps | Idea | Link |
|---|---|---|---|
| DPM-Solver++ / UniPC | 10–25 | High-order ODE solvers | [arXiv:2211.01095](https://arxiv.org/abs/2211.01095) |
| Progressive distillation | 4–8 | Student matches 2 teacher steps with 1, repeated | [arXiv:2202.00512](https://arxiv.org/abs/2202.00512) |
| Consistency models / sCM | 1–2 | Map any point on the ODE trajectory to its endpoint | [CM](https://arxiv.org/abs/2303.01469), [sCM](https://arxiv.org/abs/2410.11081) |
| Adversarial distillation (ADD, DMD2) | 1–4 | GAN-style or distribution-matching loss on the student | [ADD](https://arxiv.org/abs/2311.17042), [DMD2](https://arxiv.org/abs/2405.14867) |
| Shortcut models | 1–N | Condition on step size; learn self-consistency | [arXiv:2410.12557](https://arxiv.org/abs/2410.12557) |
| **MeanFlow** | 1 | Learn the *average* velocity over an interval: one-step generation from scratch | [arXiv:2505.13447](https://arxiv.org/abs/2505.13447) |

## 5. Guidance

- **CFG** is used by nearly every deployed model. Use guidance scales of 3–7 for images. Too high gives saturated, low-diversity samples.
- **Guidance only in a limited interval** of noise levels improves FID and diversity ([Kynkäänniemi et al. 2024](https://arxiv.org/abs/2404.07724)).
- **Autoguidance** guides with a *weaker version of the same model* instead of the unconditional model ([Karras et al. 2024](https://arxiv.org/abs/2406.02507)).

## 6. Evaluation

- **FID** is the standard but is biased by its Inception features. **FD-DINOv2** is increasingly reported alongside it.
- Use CLIP score or VQA-based metrics for text alignment, and human preference arenas for the final decision.
- Report: number of samples (50k), guidance scale, sampler, number of function evaluations (NFE), and whether EMA weights were used.
- For a holistic DiT benchmark see [DiffusionBench](https://arxiv.org/abs/2606.24888) (2026).

## 7. Diffusion beyond images

- **Video:** latent video DiTs with 3-D RoPE (Movie Gen, Wan, HunyuanVideo, Veo, Cosmos). See [02-world-models.md](02-world-models.md) for the interactive versions.
- **Text:** masked/discrete diffusion LMs. See [01-language-models.md §7](01-language-models.md#7-beyond-the-autoregressive-transformer).
- **Science:** protein structure (AlphaFold 3 uses a diffusion module), materials, weather (GenCast).
- **Policies:** Diffusion Policy and flow-matching action heads in robot VLAs such as π0.

## 8. What the code in this repo teaches

- `DiT` contains: patchify conv, fixed 2-D sincos positions, sinusoidal t-embedding, a label embedding with an extra "null" class, `DiTBlock` with adaLN-Zero, QK-norm and SwiGLU, and a zero-initialised final layer.
  - `test_dit_shapes_zero_init_and_sampling` checks that the untrained network outputs exactly zero, which is the adaLN-Zero property.
- `FlowMatching` contains: logit-normal t, straight-path interpolation, velocity target, v- or x-prediction, Euler/Heun with SD3 time-shift, CFG in one batched forward pass, and trajectory export.
- `train.py` adds: EMA weights, label dropout (needed for CFG), warmup, gradient clipping, sample grids every `--sample-every` steps.
- `sample.py --eval --cfg-sweep 1 2 3 5` trains a small classifier on real data and reports **conditional accuracy versus diversity** as a function of guidance. This is the fidelity–diversity trade-off in one table.

## 9. Exercises

1. Compare `--t-sampling uniform` vs `logit_normal` at equal steps.
2. Sample with `--sampler euler --steps 4/8/16` vs `heun`. Plot conditional accuracy against NFE (Heun uses 2 NFE per step).
3. Train `--prediction x` vs `v` with `--patch-size 8`. Which degrades less?
4. Implement a 2-step distillation: train a student to match two Heun steps of the teacher in one step.
5. Implement autoguidance by guiding with an early checkpoint of the same run.
