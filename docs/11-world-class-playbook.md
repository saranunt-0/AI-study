# 11 · Playbook: Building a World-Class AI Solution

> A practical checklist that ties the previous chapters together. It is written for a small, strong team that wants frontier-quality results on a focused problem, not for a 1,000-GPU lab.

## Short conclusion

1. **Start from the problem and the evaluation, not the model.** Define success metrics, failure costs and the evaluation set first.
2. **Beat strong baselines before you build.** An API model with good prompting and retrieval, or a fine-tuned open model, is the bar. Most "world-class" products are excellent *systems* around a model.
3. **Own the data engine.** Collect, filter, label, synthesise and verify. Data quality and coverage dominate architecture.
4. **Copy proven recipes, then ablate.** Use the defaults in chapters 01–05. Change one thing at a time, measure noise, and scale only what wins on a scaling curve.
5. **Engineer for reliability.** Reproducibility, monitoring, atomic checkpoints, CI tests of model code (like `tests/` here), and stable training (chapter 05).
6. **Post-train for the actual task** with verifiable rewards where possible, and red-team the rewards.
7. **Ship with safety and observability.** Evaluate dangerous capabilities and failure modes, monitor production, and keep humans in the loop where errors are costly.

---

## Phase 0: framing (1–2 weeks)

- [ ] Write the one-paragraph problem statement and **who is harmed by errors**.
- [ ] Define 1 primary metric + 3–5 guardrail metrics (latency, cost, safety, fairness).
- [ ] Build a **held-out, decontaminated evaluation set** with clear grading; version it.
- [ ] Estimate the budget: compute (GPU-hours), data acquisition, people, inference cost per request.
- [ ] Read the 3–5 most relevant recent papers *and* their code.

## Phase 1: baselines (1–3 weeks)

- [ ] Frontier API + prompt engineering + retrieval (RAG) baseline.
- [ ] Open-weights model (Qwen, DeepSeek, Llama, Gemma, gpt-oss families) zero-shot and fine-tuned (LoRA/QLoRA).
- [ ] Error analysis: read 100 failures by hand and categorise them. **This step is where most teams under-invest.**

## Phase 2: data engine

- [ ] Pipelines for sourcing, dedup, quality filtering, PII removal, decontamination (chapter 04 §1).
- [ ] Synthetic data with **verification** (unit tests, solvers, cross-model agreement).
- [ ] Data cards: provenance, licence, known biases.

## Phase 3: training (if needed)

- [ ] Golden config from a proven recipe; **overfit-one-batch test**; initial loss ≈ ln(V).
- [ ] Small-scale ablations with ≥2 seeds; µP/CompleteP for hyper-parameter transfer.
- [ ] Scaling-law fit before the big run; WSD schedule so you can branch decays.
- [ ] Stability: QK-norm, z-loss, clipping, spike guard, rewind plan (chapter 05).
- [ ] Throughput: MFU target ≥40%; profile before optimising (chapter 04 §7).

## Phase 4: post-training

- [ ] SFT on curated demonstrations → preference optimisation → RLVR on verifiable sub-tasks (chapter 06).
- [ ] Reward-hacking audits; held-out evaluations that are not the training reward.

## Phase 5: inference and deployment

- [ ] Serving engine (vLLM/SGLang/TensorRT-LLM), quantisation, speculative decoding, caching (chapter 07).
- [ ] Load testing; p50/p95 latency; cost per 1k requests.
- [ ] Monitoring: input drift, output quality sampling, user feedback loops, incident response.

## Phase 6: safety, ethics, governance

- [ ] Threat model (misuse, malfunction, systemic); red-team; document mitigations (chapter 09).
- [ ] Applicable regulation (EU AI Act risk tier, sector rules) and privacy review.
- [ ] Model and system cards; an honest statement of limitations.

## Habits of world-class researchers and engineers

- **Read primary sources and code**, not summaries. Re-derive key equations (the five lines in chapter 03 §1).
- **Keep a lab notebook** (a markdown run log): hypothesis → change → result → decision.
- **Prefer simple, scalable methods** (the Bitter Lesson), but measure inductive biases honestly (chapter 10 §5).
- **Grade your evidence**: established, strong, emerging, opinion, speculation (as this knowledge base tries to do).
- **Stay current efficiently:**
  - [Hugging Face Papers](https://huggingface.co/papers) (daily);
  - [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent) / [cs.CL](https://arxiv.org/list/cs.CL/recent);
  - lab blogs (Anthropic, DeepMind, Meta FAIR, Qwen, DeepSeek, Moonshot, NVIDIA);
  - [Epoch AI](https://epoch.ai/) for trends;
  - [Interconnects](https://www.interconnects.ai/) (post-training);
  - [Ahead of AI](https://magazine.sebastianraschka.com/) (architectures);
  - [Import AI](https://importai.substack.com/) (policy and research).
