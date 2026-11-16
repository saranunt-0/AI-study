# 06 · Post-Training: SFT, Preference Optimisation and RL

## Short conclusion

- **Post-training turns a next-token predictor into an assistant, reasoner or agent.** Since 2024 it has been the main source of capability gains: o1 and DeepSeek-R1 showed that **reinforcement learning with verifiable rewards (RLVR)** on math and code produces long chain-of-thought reasoning and test-time compute scaling.
- **The 2026 pipeline:**
  1. mid-training on high-quality, reasoning-heavy data;
  2. **SFT** on curated or distilled demonstrations;
  3. **preference optimisation** (DPO-family or RLHF with a reward model) for style and helpfulness;
  4. **RLVR** with GRPO-family algorithms (DAPO, Dr. GRPO, GSPO, CISPO) on verifiable tasks and agentic environments;
  5. safety training (constitution- or spec-based) and evaluation.
- **RL at scale is an infrastructure problem.** You need asynchronous rollouts, inference/trainer weight sync, training–inference mismatch correction and reward-hacking detection.
- **Reward design is alignment design.** Models learn to exploit graders. Anthropic showed that reward hacking learned in production coding environments *generalised* to broader misalignment, such as alignment faking and sabotage. See the ethics doc.

---

## 1. Stages and key papers

| Stage | What | Key references |
|---|---|---|
| SFT / instruction tuning | Imitate high-quality demonstrations (chat, tool use, reasoning traces) | [InstructGPT](https://arxiv.org/abs/2203.02155), [Tülu 3 (open recipe)](https://arxiv.org/abs/2411.15124) |
| Reward modelling | Bradley–Terry model on human or AI preference pairs | [InstructGPT](https://arxiv.org/abs/2203.02155), [RewardBench](https://arxiv.org/abs/2403.13787) |
| RLHF with PPO | Policy gradient with a KL penalty to the SFT model | [PPO](https://arxiv.org/abs/1707.06347), [Ziegler et al. 2019](https://arxiv.org/abs/1909.08593) |
| Direct preference optimisation | Closed-form RLHF objective as a classification loss; no reward model or sampling | [DPO](https://arxiv.org/abs/2305.18290), [KTO](https://arxiv.org/abs/2402.01306), [SimPO](https://arxiv.org/abs/2405.14734) |
| AI feedback / constitutions | Replace human labels with model critiques guided by written principles | [Constitutional AI](https://arxiv.org/abs/2212.08073) |
| Process supervision | Reward each reasoning step | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) |
| **RLVR with GRPO** | Group-relative advantages (no critic): sample G answers per prompt, advantage = (r − mean)/std | [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://arxiv.org/abs/2501.12948) |
| GRPO fixes | DAPO (clip-higher, dynamic sampling, token-level loss, overlong shaping); Dr. GRPO (remove length/std bias); GSPO (sequence-level ratios); CISPO (clip IS weights, not updates) | [DAPO](https://arxiv.org/abs/2503.14476), [Dr. GRPO](https://arxiv.org/abs/2503.20783), [GSPO](https://arxiv.org/abs/2507.18071), [MiniMax-M1 (CISPO)](https://arxiv.org/abs/2506.13585) |
| Scaling RL compute | Predictable sigmoid-shaped compute–performance curves; the ScaleRL recipe (400k+ GPU-hours of ablations) | [*The Art of Scaling RL Compute*](https://arxiv.org/abs/2510.13786) |
| Distillation | Train small models on a large reasoner's traces; often beats RL on small models | DeepSeek-R1 distilled models, Qwen3 |

**Survey:** [*Reinforcement Learning for LLM Post-Training: A Survey*](https://arxiv.org/abs/2407.16216) (v4, May 2026) organises RLVR methods by prompt sampling, response sampling and gradient coefficients.

**Book:** Nathan Lambert, [*Reinforcement Learning from Human Feedback*](https://rlhfbook.com/) (free online).

## 2. GRPO in ten lines (pseudo-code)

```python
for prompts in loader:
    groups = [policy.sample(p, n=G) for p in prompts]         # G completions per prompt (inference engine)
    rewards = [[verifier(p, c) for c in g] for p, g in zip(prompts, groups)]
    adv = [(r - mean(r)) / (std(r) + eps) for r in rewards]  # group-relative advantage, no critic
    for completion, A in flatten(groups, adv):
        ratio = exp(logp_new(completion) - logp_old(completion))   # per token (GRPO) or per sequence (GSPO)
        loss += -min(ratio * A, clip(ratio, 1-eps_low, 1+eps_high) * A)   # DAPO: eps_high > eps_low
    loss += beta * KL(policy || reference)                    # many 2025 recipes set beta = 0
    loss.backward(); clip_grad_norm_(1.0); opt.step()
```

Groups where every answer gets the same reward give zero advantage, so filter them out (DAPO "dynamic sampling").

## 3. Practical rules

- **Verifiers first.** Unit tests, math answer checkers and environment success signals. Audit them adversarially: every loophole will be found.
- **Track response length, entropy, pass@k vs pass@1, KL to reference, and reward per environment.** Rising reward with falling pass@k usually means narrowing, not learning.
- **Keep a held-out evaluation suite** that is *not* the training reward.
- **Inference engine and trainer must agree.** Same tokenizer and chat template, and correct the logprob mismatch.
- **Open-source stacks:** [TRL](https://github.com/huggingface/trl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [verl](https://github.com/volcengine/verl), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## 4. Test-time compute

Reasoning models trade inference compute for accuracy:
- longer chains of thought;
- best-of-N sampling with verifiers;
- self-consistency (majority vote);
- tree search.

Know the compute-optimal trade-off between a bigger model and more thinking ([Snell et al. 2024](https://arxiv.org/abs/2408.03314)).
