# 10 · Neuroscience and AI: Connectomes, Fly Brains and NeuroAI

## Short conclusion

- **2024–26 moved neuroscience from wiring diagrams to running models of whole nervous systems.**
  - The adult fruit-fly brain connectome (FlyWire, Nature 2024: ~139k neurons, ~50M synapses) is complete.
  - A connectome-based spiking model *predicted* real feeding and grooming circuits.
  - Connectome-constrained networks predicted visual responses across 26 experiments.
  - In March 2026 Eon Systems connected a whole-brain fly model to a physics-simulated body (NeuroMechFly) and reported walking, grooming and feeding behaviours. This is **not yet peer reviewed**, and experts dispute what it shows.
- **The flow of ideas runs both ways:**
  - **AI → neuroscience:** deep nets are the best predictive models of visual cortex; connectome + gradient descent can infer unknown neuron parameters; ML segments connectomes.
  - **neuroscience → AI:** reinforcement learning (dopamine = TD error), convolution and attention analogies, replay, predictive coding and world models (the free-energy principle, JEPA), sparse coding, and the fly olfactory circuit as a locality-sensitive hash.
- **What brains still do far better:**
  - energy efficiency (~20 W);
  - continual learning without catastrophic forgetting;
  - sample efficiency;
  - robust sensorimotor control from tiny networks (a fly does flight control, navigation, courtship and learning with ~140k neurons).
  
  These are the gaps to mine for ideas.
- **Caution.** "We uploaded a fly" headlines overstate the case. Connectomes give structure, not the full dynamics: neuromodulators, gap junctions, plasticity and many synaptic weights or signs are still estimated.

---

## 1. The fruit fly (*Drosophila melanogaster*) as a model system

| Result | What was shown | Link |
|---|---|---|
| **FlyWire whole-brain connectome** (Dorkenwald et al., *Nature* Oct 2024) | Complete wiring diagram of an adult female fly brain: 139,255 neurons, ~50M synapses; 10-year consortium; AI-assisted segmentation + crowd proofreading | [Nature](https://doi.org/10.1038/s41586-024-07558-y), [flywire.ai](https://flywire.ai/), companion cell-typing paper: [Schlegel et al.](https://doi.org/10.1038/s41586-024-07686-5) |
| **Whole-brain leaky integrate-and-fire model** (Shiu et al., *Nature* 2024) | Connectivity + neurotransmitter identity alone → a spiking model that predicted which neurons drive feeding (proboscis extension) and grooming; predictions confirmed in real flies. Runs on a laptop | [Nature](https://doi.org/10.1038/s41586-024-07763-9), [Berkeley news](https://vcresearch.berkeley.edu/news/researchers-simulate-fly-brain-laptop-human-brain-next) |
| **Connectome-constrained deep networks** (Lappalainen et al., *Nature* 2024) | 64 cell types of the visual motion pathway; connectivity from the connectome, unknown single-neuron/synapse parameters fit by deep learning on a motion task → predicted neural activity across 26 experimental studies | [Nature](https://doi.org/10.1038/s41586-024-07939-3) |
| **NeuroMechFly v2** (Wang-Chen et al., *Nature Methods* 2024) | Physics-simulated fly body (MuJoCo) with vision, olfaction, leg adhesion and ascending feedback; odour-plume navigation, fly following | [Nature Methods](https://doi.org/10.1038/s41592-024-02497-y) |
| **Connectome graph model for locomotion** (Feb 2026 preprint) | Uses the whole-brain connectome as the *architecture* of a controller trained by RL for whole-body locomotion | [arXiv:2602.17997](https://arxiv.org/abs/2602.17997) |
| **Eon Systems embodied whole-brain emulation** (Mar 2026, company announcement) | FlyWire-based LIF brain (Shiu-style) closed-loop with NeuroMechFly v2: the simulated fly walked, groomed its antennae, and extended its proboscis toward a sugar-like signal. **Not peer reviewed**; critics note the simulated brain does not operate as a real fly's does; reported neuron counts differ between sources. Next goal: mouse | [The Register](https://www.theregister.com/2026/03/16/digital_fruit_fly_brain_model/), [code: eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) |
| **Fly olfaction as a hashing algorithm** (Dasgupta, Stevens, Navlakha, *Science* 2017) | The fly's odour circuit performs sparse, *expanding* random projection + winner-take-all, a locality-sensitive hash that beat classic LSH on similarity search | [Science](https://doi.org/10.1126/science.aam9868) |

**Why a fly matters for AI people.** It is the largest complete brain we can both *map* and *simulate*, and it solves embodied control, navigation and learning with ~140k neurons and microwatts. That is a hard benchmark for "intelligence per joule" and for inductive biases that work with tiny data.

## 2. Mammalian and human connectomics

| Project | Scale | Link |
|---|---|---|
| **MICrONS** (Allen Institute, Baylor, Princeton; *Nature* Apr 2025) | 1 mm³ of mouse visual cortex: ~200k cells, ~500M synapses, ~4 km of axons, with **functional recordings of ~75k neurons** while the mouse watched videos; 1.6 PB of public data | [Nature](https://doi.org/10.1038/s41586-025-08790-w), [MICrONS Explorer](https://www.microns-explorer.org/) |
| **H01** (Harvard/Google, *Science* 2024) | 1 mm³ of human temporal cortex at synaptic resolution | [Science](https://doi.org/10.1126/science.adk4858) |
| NIH BRAIN Initiative NeuroAI | 2026 NeuroAI roadmap sessions; MICrONS analyses (e.g. CMU/Pitkow) | [CMU news (2026)](https://www.cmu.edu/mcs/news-events/2026/0806-pitkow-nature) |

## 3. Theory bridges: brain ideas that shaped AI (and back)

| Idea | Neuroscience | AI counterpart | Link |
|---|---|---|---|
| Reward prediction error | Dopamine neurons fire like a TD error | TD learning, actor–critic, RLHF/RLVR | [Schultz, Dayan, Montague 1997](https://doi.org/10.1126/science.275.5306.1593) |
| Distributional value | Dopamine neurons encode a *distribution* of future reward | Distributional RL | [Dabney et al., *Nature* 2020](https://doi.org/10.1038/s41586-019-1924-6) |
| Predictive coding / free energy | Cortex predicts its inputs; errors propagate up | World models, JEPA, active inference | [Rao & Ballard 1999](https://doi.org/10.1038/4580), [Friston 2010](https://doi.org/10.1038/nrn2787) |
| Hierarchical visual processing | Ventral stream V1→V4→IT | CNNs are the best predictive models of IT | [Yamins & DiCarlo 2016](https://doi.org/10.1038/nn.4244) |
| Hippocampus, grid and place cells | Cognitive maps, replay | Grid-like codes emerge in RNNs; Transformers ≈ hippocampal models (TEM) | [Banino et al. 2018](https://doi.org/10.1038/s41586-018-0102-6), [Whittington et al. 2020](https://doi.org/10.1016/j.cell.2020.10.024), [Transformers & hippocampus](https://arxiv.org/abs/2112.04035) |
| Complementary learning systems / replay | Fast hippocampal + slow cortical learning | Experience replay, continual learning | [Hassabis et al., *Neuroscience-Inspired AI*](https://doi.org/10.1016/j.neuron.2017.06.011) |
| Credit assignment | Does the brain do backprop? | Feedback alignment, predictive-coding approximations | [Lillicrap et al. 2020](https://doi.org/10.1038/s41583-020-0277-3) |
| Cortical columns as sensorimotor models | "Thousand Brains" theory | Open-source sensorimotor learning framework | [Thousand Brains Project](https://thousandbrains.org/) |
| Embodiment | Intelligence is shaped by the body and its environment | Robotics foundation models, world models | [Zador et al., *Catalyzing next-generation AI through NeuroAI*](https://doi.org/10.1038/s41467-023-37180-x) |

## 4. Biological and neuromorphic computing

- **DishBrain** (Kagan et al., *Neuron* 2022): ~800k cultured human and mouse neurons on a multi-electrode array learned to play Pong in a closed loop ([paper](https://doi.org/10.1016/j.neuron.2022.09.001)).
- **Cortical Labs CL1** (Mar 2025): a commercial biological computer with lab-grown human neurons on silicon, available remotely ("wetware-as-a-service"). It reportedly played DOOM in 2026. Neuron-count claims vary by source ([corticallabs.com](https://corticallabs.com/)).
- **Organoid intelligence**: a research programme on brain organoids as computing substrates ([Smirnova et al. 2023](https://doi.org/10.3389/fsci.2023.1017235)). It raises serious ethical questions about sentience in vitro.
- **Neuromorphic chips** (Intel Loihi 2, SpiNNaker 2, IBM NorthPole): event-driven, co-located memory and compute, aimed at the brain's energy efficiency.

## 5. Lessons for AI builders

1. **Inductive biases from biology are cheap priors.** Connectome-constrained models needed far less data to predict neural activity than unconstrained ones (Lappalainen et al.). Structure substitutes for data.
2. **Sparsity and expansion** (fly olfaction, cortex) map to MoE, sparse attention and wide hidden layers: compute only what you need.
3. **Prediction is the core objective** across theories (predictive coding, free energy, JEPA, next-token prediction). The disagreement is *what* to predict: pixels, tokens or abstract states.
4. **Embodiment and closed loops matter.** The Eon demo and NeuroMechFly show that behaviour emerges from the brain, body and environment together, echoing Bennett's critique of "computational dualism" (see [09-ai-philosophy-and-ethics.md §3](09-ai-philosophy-and-ethics.md#3-michael-timothy-bennett-weakness-stack-theory-and-conscious-machines)).
5. **Energy is the next frontier metric.** The brain runs on ~20 W. Report joules per token or per decision alongside accuracy.

## 6. Learning resources

- Neuromatch Academy (free computational neuroscience and NeuroAI courses): [neuromatch.io](https://neuromatch.io/)
- Dayan & Abbott, *Theoretical Neuroscience*; Gerstner et al., [*Neuronal Dynamics*](https://neuronaldynamics.epfl.ch/) (free online)
- FlyWire Codex for browsing the fly connectome: [codex.flywire.ai](https://codex.flywire.ai/)
- Hands-on: run the [Shiu et al. LIF model](https://github.com/philshiu/Drosophila_brain_model) and the [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym) simulator
