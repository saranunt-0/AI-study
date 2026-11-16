# 02 · World Models

> **Code:** [`aistudy/world_model/`](../aistudy/world_model): `env.py` (TwoRoom pixel environment) · `model.py` (JEPA world model + SIGReg + CEM planner) · `train.py` · `plan.py` (inference = goal-conditioned model-predictive control)

## Short conclusion

- A **world model** learns how the world changes in response to actions: `p(s_{t+1} | s_≤t, a_t)`. You can use it to *simulate*, *plan* or *train agents in imagination*.
- In 2026 the field splits into three families with different goals:
  1. **Generative, interactive video models** (DeepMind Genie 3 → Genie 4, NVIDIA Cosmos, GameNGen, DIAMOND, World Labs Marble). They produce photorealistic pixels for content, simulation and synthetic data. They are expensive to plan with and their physics is brittle.
  2. **Latent predictive models (JEPA)** (Meta V-JEPA 2 / 2-AC, DINO-WM, PLDM, LeJEPA → LeWorldModel, and LeCun's AMI Labs). They predict in representation space, are cheap to plan with, and are aimed at robotics and agents.
  3. **RL world models** (Dreamer V1–V3 → Dreamer 4, MuZero, TD-MPC2). The model exists to train or plan a policy. Dreamer 4 obtained diamonds in Minecraft purely from offline data.
- **The open debate.** One side argues pixels are needed (generative). The other argues that predicting abstract, controllable features is enough and more efficient (LeCun). Both sides have strong 2025–26 results. Expect hybrids: generative models for training data, latent models for control.
- **Practical takeaway:**
  - evaluate world models by **what they enable** (planning success, policy return, downstream probes), not by pixel quality;
  - data coverage of the action space is the main bottleneck;
  - compounding error over long rollouts is the main failure mode.

---

## 1. Concepts you must own

| Concept | Meaning | Where it shows up |
|---|---|---|
| Observation encoder | Pixels → latent state `z_t` | Every model; frozen DINOv2 in DINO-WM, learned in LeWM |
| Dynamics / predictor | `z_{t+1} = f(z_≤t, a_t)`, deterministic or stochastic | RSSM (Dreamer), Transformer (Genie, V-JEPA 2-AC), diffusion (DIAMOND) |
| Representation collapse | A latent-only objective is trivially minimised by a constant `z` | Fixed by reconstruction, EMA teacher + stop-grad (BYOL/V-JEPA), VICReg, or **SIGReg** (LeJEPA) |
| Compounding error | Small one-step errors explode over long open-loop rollouts | Fix with multi-step rollout losses, noise augmentation, re-planning (MPC) |
| Action conditioning | Inject `a_t` by concatenation, AdaLN, or cross-attention | Genie learns *latent actions* from unlabeled video |
| Planning | Search action sequences with the model: CEM, MPPI, MCTS, gradient-based | MuZero (MCTS), TD-MPC2 (MPPI), DINO-WM / V-JEPA 2-AC / LeWM (CEM) |
| Imagination training | Train an actor-critic purely on model rollouts | Dreamer family |
| Memory & consistency | Revisiting a place should reproduce it | The main weakness of video world models; Genie 3 was the first with minutes of consistency |

## 2. The landscape (verified primary sources where possible)

### 2a. Generative interactive world models
| Model | Year | Key idea | Link |
|---|---|---|---|
| World Models (Ha & Schmidhuber) | 2018 | VAE + MDN-RNN + controller trained "in the dream" | [arXiv:1803.10122](https://arxiv.org/abs/1803.10122), [interactive](https://worldmodels.github.io/) |
| Genie | 2024 | Learns **latent actions** from unlabeled 2D game video | [arXiv:2402.15391](https://arxiv.org/abs/2402.15391) |
| GameNGen | 2024 | Diffusion model runs DOOM in real time | [arXiv:2408.14837](https://arxiv.org/abs/2408.14837) |
| DIAMOND | 2024 | Diffusion world model as an RL environment (Atari) | [arXiv:2405.12399](https://arxiv.org/abs/2405.12399) |
| NVIDIA Cosmos | 2025 | Open "world foundation model" platform for physical AI (tokenizers, diffusion and AR models). **Cosmos 3** (June 2026): an omni-model family (Nano 16B / Super 64B) unifying understanding, generation and action | [arXiv:2501.03575](https://arxiv.org/abs/2501.03575), [GitHub](https://github.com/nvidia-cosmos) |
| Genie 3 | Aug 2025 | Real-time (24 fps, 720p) text-prompted interactive worlds with minutes of consistency. Waymo built a driving world model on it (Feb 2026) | [DeepMind blog](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) |
| Genie 4 | Oct 27 2026 (research preview) | Adds generated audio and mid-session steering. Known limit: drift after a few minutes. *Coverage is mostly secondary; check the DeepMind post* | [coverage](https://www.emergentmind.com/genie-4) |
| World Labs Marble | 2025 | Persistent, exportable 3D worlds (Gaussian splats) instead of frames | [worldlabs.ai](https://www.worldlabs.ai/) |

### 2b. Latent predictive / JEPA world models
| Model | Year | Key idea | Link |
|---|---|---|---|
| JEPA position paper (LeCun) | 2022 | Predict in representation space; energy-based; hierarchical planning | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) |
| I-JEPA / V-JEPA | 2023–24 | Masked latent prediction for images / video | [I-JEPA](https://arxiv.org/abs/2301.08243), [V-JEPA](https://arxiv.org/abs/2404.08471) |
| DINO-WM | 2024 | Frozen DINOv2 patch features + learned predictor + CEM: zero-shot planning | [arXiv:2411.04983](https://arxiv.org/abs/2411.04983) |
| V-JEPA 2 / 2-AC | Jun 2025 | Pretrained on >1M hours of video. The action-conditioned version is post-trained on ~62 h of robot video (DROID) and does zero-shot pick-and-place with image goals on Franka arms | [arXiv:2506.09985](https://arxiv.org/abs/2506.09985), [code](https://github.com/facebookresearch/vjepa2) |
| PLDM | 2025 | Planning with latent dynamics from reward-free offline data; Two-Rooms benchmark | [arXiv:2502.14819](https://arxiv.org/abs/2502.14819) |
| **LeJEPA / SIGReg** | Nov 2025 | Provably optimal embedding distribution = isotropic Gaussian. The SIGReg loss enforces it, so no stop-grad, EMA teacher or other heuristics are needed | [arXiv:2511.08544](https://arxiv.org/abs/2511.08544), [code](https://github.com/galilai-group/lejepa) |
| **LeWorldModel (LeWM)** | Mar 2026 | End-to-end JEPA from pixels with prediction + SIGReg; ~15M params; trains on 1 GPU in hours; plans up to 48× faster than foundation-model world models; one loss hyper-parameter | [arXiv:2603.19312](https://arxiv.org/abs/2603.19312) |
| V-JEPA 2.1 | Mar 2026 | Dense features for video SSL (reported in surveys) | [arXiv:2603.14482](https://arxiv.org/abs/2603.14482) |
| AMI Labs (LeCun) | Mar 2026 | $1.03B seed round to build JEPA-style world models; healthcare and robotics partners | [SiliconANGLE coverage](https://siliconangle.com/2026/03/10/yann-lecuns-new-startup-ami-labs-raises-1-03b-train-world-models/) |

### 2c. World models for reinforcement learning
| Model | Year | Key idea | Link |
|---|---|---|---|
| PlaNet / Dreamer V1–V2 | 2019–20 | Recurrent State-Space Model (RSSM), latent imagination | [PlaNet](https://arxiv.org/abs/1811.04551), [DreamerV2](https://arxiv.org/abs/2010.02193) |
| MuZero | 2019 | Value-equivalent model (predicts reward/value/policy, not pixels) + MCTS | [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) |
| DreamerV3 | 2023 / Nature 2025 | One hyper-parameter set across 150+ tasks; symlog, two-hot, free bits | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104), [code](https://github.com/danijar/dreamerv3) |
| IRIS / Δ-IRIS | 2022–24 | Transformer over discrete tokens as the world model | [arXiv:2209.00588](https://arxiv.org/abs/2209.00588) |
| TD-MPC2 | 2023 | Decoder-free latent model + MPPI planning; scales to 317M params across 80 tasks | [arXiv:2310.16828](https://arxiv.org/abs/2310.16828) |
| **Dreamer 4** | Sep 2025 | Fast Transformer world model with a "shortcut forcing" objective (real-time on one GPU). First agent to get diamonds in Minecraft **purely from offline data**, using 100× less data than VPT | [arXiv:2509.24527](https://arxiv.org/abs/2509.24527), [project](https://danijar.com/project/dreamer4/) |

**Tutorials and surveys:** [*From World Models to World Action Models: A Concise Tutorial for Robotics*](https://arxiv.org/abs/2607.00836) (2026) · [Wikipedia: World model (AI)](https://en.wikipedia.org/wiki/World_model_(artificial_intelligence)) · [Lilian Weng on model-based RL](https://lilianweng.github.io/)

## 3. How to choose an approach

| You need… | Choose | Why |
|---|---|---|
| Photorealistic simulation, content, synthetic training data | Generative video world model (Cosmos, Genie-style) | Pixel fidelity is the product |
| Robot or agent planning from images with little action data | JEPA latent model + CEM/MPPI (V-JEPA 2-AC, DINO-WM, LeWM) | Cheap rollouts; no pixel generation in the loop |
| Maximum task return in a simulator or game | Dreamer / TD-MPC2 style | Model and policy co-trained for reward |
| Board games / perfect simulators | MuZero / AlphaZero + MCTS | Search beats imitation |

## 4. What the code in this repo teaches

`aistudy/world_model` reproduces the LeWorldModel recipe at toy scale:

1. **Data.** `collect_random_trajectories` produces random, temporally persistent actions in the TwoRoom environment, with some starts near the door. Without door transitions in the data, the model cannot learn to plan through it.
2. **Model.**
   - CNN encoder producing `z`;
   - causal-Transformer predictor (the *same block* as the LLM) predicting `ẑ_{t+1} = z_t + f(z_≤t, a_≤t)`;
   - decoder trained on `z.detach()` as a visual *probe* only.
3. **Loss.** Teacher-forced prediction + multi-step open-loop rollout + `λ·SIGReg(z)`. `--objective recon` swaps SIGReg for pixel reconstruction (the Dreamer / World Models route) so you can compare the two.
4. **Inference.** `plan.py` encodes the goal image, runs CEM over action sequences in latent space, executes the first action, and re-plans (MPC).

**Results from one CPU run** (400 steps, ~95 s, λ = 1; your numbers will vary):

| Metric | Value |
|---|---|
| Linear-probe R² of the true (x, y) from `z` | 0.999 |
| Open-loop latent error after 16 steps (normalised) | 0.10 |
| Planning success, random goals | CEM 100% · greedy oracle 70% · random 10% |
| Planning success, goal in the other room | CEM 100% · greedy oracle 70% · random 0% |

The first version of this code used λ = 0.1 with a mean-normalised SIGReg and **collapsed**: SIGReg stuck at 0.40 and probe R² ≈ 0 while the prediction loss looked "excellent" (1e-7). That is the textbook JEPA failure. A tiny prediction loss is *not* evidence of a good world model, so always log a collapse metric and a probe.

## 5. Exercises

1. Train with `--objective recon` and compare probe R², rollout error and planning success with the JEPA run.
2. Set `--sigreg-weight 0` and watch the collapse happen. Then try `--stop-grad-target` alone (BYOL-style). Does it collapse here?
3. Reduce `--rollout-steps` to 1 (teacher forcing only) and measure `rollout_nmse@16` and planning success.
4. Remove the door-biased starts (`door_frac=0` in `collect_random_trajectories`) and test `--cross-room` planning.
5. Replace CEM with MPPI (softmax-weighted average of all samples instead of top-k elites).
