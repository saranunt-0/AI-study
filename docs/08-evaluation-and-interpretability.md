# 08 · Evaluation and Interpretability

## Short conclusion

- **You cannot improve what you cannot measure.** Build the evaluation suite before the model.
- **Public benchmarks saturate in 1–2 years and leak into training data**, so pair them with private, decontaminated, task-specific evaluations.
- **Prefer evaluations that measure what you actually want:**
  - agentic task completion (SWE-bench Verified, METR time horizons);
  - adaptive reasoning on novel problems (ARC-AGI-3, where frontier models scored <1% at launch in March 2026);
  - expert-level knowledge (GPQA, Humanity's Last Exam).
- **Interpretability moved from toy models to production models** (sparse autoencoders → attribution graphs on Claude 3.5 Haiku). It is now used for safety audits, debugging and understanding emergent behaviours such as introspection.

---

## 1. Evaluation

| Category | Benchmarks | Notes |
|---|---|---|
| Knowledge / reasoning | [MMLU-Pro](https://arxiv.org/abs/2406.01574), [GPQA](https://arxiv.org/abs/2311.12022), [Humanity's Last Exam](https://arxiv.org/abs/2501.14249) | Saturation and contamination risk |
| Math | AIME (by year), [FrontierMath](https://arxiv.org/abs/2411.04872), IMO problems | Gold-medal IMO performance reached in 2025 |
| Code / agents | [SWE-bench Verified](https://www.swebench.com/), [LiveCodeBench](https://arxiv.org/abs/2403.07974), Terminal-Bench | Agent scaffolding matters as much as the model |
| Long-horizon autonomy | [METR: time horizon of tasks AI can complete](https://arxiv.org/abs/2503.14499) | Horizon doubling roughly every ~7 months (2019–2025 trend) |
| Fluid intelligence | [ARC-AGI-2 / ARC-AGI-3](https://arcprize.org/) | ARC-AGI-3 (Mar 2026) is interactive: no instructions, agents must explore, model, set goals and plan. All frontier models scored <1% at launch while humans solved all environments |
| Human preference | [LMArena](https://lmarena.ai/) | Style-sensitive; use for UX, not capabilities |
| Safety | Dangerous-capability evals (bio, cyber, autonomy), jailbreak robustness | See the frontier safety frameworks in the ethics doc |

**Rules of thumb:**
1. Report confidence intervals: bootstrap over questions, plus variance over seeds and sampling.
2. Fix prompts, chat templates, temperature and the answer-extraction code, and publish them.
3. Decontaminate with n-gram overlap checks against training data.
4. Validate LLM-as-judge against human labels before trusting it.
5. Small-scale proxies: for pre-training ablations, use evaluations that are above chance at small scale (the Smol playbook calls this "signal at small scale").

## 2. Interpretability

| Topic | What you learn | Link |
|---|---|---|
| Circuits & superposition | Networks represent more features than dimensions | [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) |
| Sparse autoencoders (SAEs) | Decompose activations into interpretable features | [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features), [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) |
| Attribution graphs / circuit tracing | Trace how a production model computes an answer (planning in poems, multi-step reasoning, hallucination circuits) | [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) |
| Probing & steering | Linear probes; activation steering vectors | [Representation Engineering](https://arxiv.org/abs/2310.01405) |
| Introspection | Evidence that models can sometimes report on their internal states | [Emergent introspective awareness (Anthropic, 2025)](https://transformer-circuits.pub/2025/introspection/index.html) |
| Training dynamics | Grokking, induction heads, phase transitions | [In-context learning and induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) |

**Start here:** the [Transformer Circuits thread](https://transformer-circuits.pub/), [Neel Nanda's resources](https://www.neelnanda.io/mechanistic-interpretability), and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

**In this repo:** the world-model **linear probe** (`probe_r2` in `world_model/train.py`) is a minimal interpretability tool. It asks whether a variable (position) is linearly readable from a representation. The diffusion **classifier-based metric** (`sample.py --eval`) is a minimal evaluation-by-recogniser, which is the idea behind FID and Inception Score.
