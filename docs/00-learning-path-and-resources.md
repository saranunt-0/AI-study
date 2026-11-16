# 00 · Learning Path and Curated Resources (Advanced)

## Short conclusion

- **Who this is for:** someone who already trains neural networks and wants to reach research-engineer level on frontier generative models.
- **The fastest path** combines three things:
  1. **one deep course per pillar** (CS336 for LLMs, MIT 6.S184 for diffusion and flow, Sutton & Barto + Dreamer/JEPA papers for world models and RL);
  2. **re-implementing from scratch** (this repo, then nanoGPT/nanochat, modded-nanogpt, DiT, DreamerV3);
  3. **reading frontier technical reports** (DeepSeek, Qwen, Kimi, OLMo, SmolLM3) with their code.
- **Budget:** about 16 weeks at 10–15 h/week. The plan below maps each week to a doc and a script in this repo.

---

## 1. Prerequisites (refresh only what you need)

| Topic | Resource |
|---|---|
| Linear algebra, probability, calculus for ML | [*Mathematics for Machine Learning*](https://mml-book.github.io/) (free) |
| Deep learning foundations | [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) (Prince, free) · [*Deep Learning: Foundations and Concepts*](https://www.bishopbook.com/) (Bishop) · [*The Little Book of Deep Learning*](https://fleuret.org/francois/lbdl.html) (Fleuret, free) · [*Dive into Deep Learning*](https://d2l.ai/) |
| Intuition | [3Blue1Brown: neural networks & transformers](https://www.3blue1brown.com/topics/neural-networks) · [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |
| From-scratch coding | [Karpathy: Neural Networks Zero to Hero](https://karpathy.ai/zero-to-hero.html) · [Raschka: LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) |

## 2. Core courses (advanced)

| Pillar | Course | Why |
|---|---|---|
| LLMs | [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) | Tokenizer → Transformer → kernels → parallelism → scaling laws → data → RL; assignments build everything |
| LLMs (systems) | [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) · [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) · [GPU MODE lectures](https://github.com/gpu-mode/lectures) | Distributed training, rooflines, CUDA/Triton |
| LLMs (practice) | [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) · [karpathy/nanochat](https://github.com/karpathy/nanochat) | End-to-end real decisions; a full ChatGPT-style pipeline for about $100 |
| Diffusion / flow | [MIT 6.S184: Flow Matching & Diffusion](https://diffusion.csail.mit.edu/) · [*The Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890) · [*Flow Matching Guide and Code*](https://arxiv.org/abs/2412.06264) | The unified theory + code |
| RL & world models | [Sutton & Barto, *Reinforcement Learning* (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html) · [OpenAI Spinning Up](https://spinningup.openai.com/) · [DreamerV3 code](https://github.com/danijar/dreamerv3) | Foundations for model-based RL |
| Post-training | [Nathan Lambert, *RLHF book*](https://rlhfbook.com/) · [Hugging Face TRL](https://github.com/huggingface/trl) | SFT, DPO, GRPO in practice |
| Interpretability | [Transformer Circuits](https://transformer-circuits.pub/) · [ARENA curriculum](https://www.arena.education/) | Mechanistic interpretability hands-on |
| NeuroAI | [Neuromatch Academy](https://neuromatch.io/) | Computational neuroscience + NeuroAI |

## 3. Blogs and writers worth following

[Lilian Weng](https://lilianweng.github.io/) (deep surveys) · [Sebastian Raschka / Ahead of AI](https://magazine.sebastianraschka.com/) (architecture comparisons) · [Sander Dieleman](https://sander.ai/) (diffusion) · [Yang Song](https://yang-song.net/blog/2021/score/) (score models) · [Keller Jordan](https://kellerjordan.github.io/posts/muon/) and [Jeremy Bernstein](https://jeremybernste.in/) (optimizers) · [Physics of Language Models](https://physics.allen-zhu.com/) · [Interconnects (Nathan Lambert)](https://www.interconnects.ai/) · [Epoch AI](https://epoch.ai/) · [Distill (archive)](https://distill.pub/) · [Stas Bekman's ML Engineering](https://github.com/stas00/ml-engineering)

## 4. Reference code to study (after this repo)

| Repo | Learn |
|---|---|
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), [nanochat](https://github.com/karpathy/nanochat) | Minimal GPT training and a full chat pipeline |
| [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | State-of-the-art small-scale training tricks (Muon, etc.) |
| [pytorch/torchtitan](https://github.com/pytorch/torchtitan) | Production PyTorch pre-training (FSDP2, TP, PP, float8) |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Frontier-scale parallelism |
| [deepseek-ai](https://github.com/deepseek-ai) | DeepSeek-V3 inference code, DeepEP, DeepGEMM, FlashMLA |
| [facebookresearch/DiT](https://github.com/facebookresearch/DiT), [flow_matching](https://github.com/facebookresearch/flow_matching), [huggingface/diffusers](https://github.com/huggingface/diffusers) | Diffusion models |
| [danijar/dreamerv3](https://github.com/danijar/dreamerv3), [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2), [galilai-group/lejepa](https://github.com/galilai-group/lejepa), [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2), [nvidia-cosmos](https://github.com/nvidia-cosmos) | World models |
| [vllm-project/vllm](https://github.com/vllm-project/vllm), [sgl-project/sglang](https://github.com/sgl-project/sglang) | Serving |

## 5. A 16-week plan mapped to this repo

| Week | Read | Do (this repo) |
|---|---|---|
| 1 | [01 §1–2](01-language-models.md), Annotated Transformer | Read `aistudy/llm/model.py` top to bottom; run `pytest -q` |
| 2 | [04 §1–4](04-training-techniques.md) | `python -m aistudy.llm.train --preset tiny`; plot `runs/llm/log.jsonl` |
| 3 | [05](05-gradient-control-and-stability.md), Wortsman et al. | Induce a spike: `--lr 3e-2 --no-qk-norm --z-loss 0`; then fix it with each tool |
| 4 | [01 §3](01-language-models.md#3-mixture-of-experts-moe), DeepSeekMoE, aux-loss-free paper | `--n-experts 8 --optimizer muon`; study `moe_max_load_ratio` |
| 5 | [04 §6–7](04-training-techniques.md#6-distributed-training), Ultra-Scale Playbook | `torchrun --nproc_per_node=2 -m aistudy.llm.train`; compute MFU on a GPU |
| 6 | [07](07-inference-and-serving.md) | `generate.py --benchmark`; speculative decoding with a draft model |
| 7 | [03 §1](03-diffusion-models.md), MIT 6.S184 lectures 1–3 | Derive the flow-matching loss; read `flow.py` |
| 8 | [03 §2–3](03-diffusion-models.md#2-architecture), DiT, SD3, JiT | `python -m aistudy.diffusion.train`; then `--prediction x --patch-size 8` |
| 9 | [03 §4–6](03-diffusion-models.md#4-fast-sampling-and-distillation) | `sample.py --eval --cfg-sweep 1 2 3 5`; implement 2-step distillation |
| 10 | [02 §1–2](02-world-models.md), Ha & Schmidhuber, DreamerV3 | Read `world_model/model.py`; train `--objective recon` |
| 11 | LeJEPA, LeWorldModel, V-JEPA 2 | Train JEPA; break it with `--sigreg-weight 0`; explain the collapse |
| 12 | DINO-WM, TD-MPC2, Dreamer 4 | `plan.py --cross-room --baselines`; implement MPPI |
| 13 | [06](06-post-training-and-alignment.md), DeepSeek-R1, DAPO | Fine-tune a small open model with TRL GRPO on a verifiable task |
| 14 | [08](08-evaluation-and-interpretability.md), Biology of an LLM | Train a linear probe / SAE on your tiny LM's activations |
| 15 | [09](09-ai-philosophy-and-ethics.md), Bennett, Chollet, Hendrycks et al. | Write a 2-page position: "what would convince me a system is AGI?" |
| 16 | [10](10-neuroscience-and-ai.md), [11](11-world-class-playbook.md) | Run the Shiu fly-brain model; write your project plan with the playbook |
