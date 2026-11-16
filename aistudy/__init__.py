"""aistudy: compact, readable reference implementations of modern generative models.

Sub-packages
------------
common       optimizers (AdamW / Muon), LR schedules, gradient control, EMA, checkpoints
llm          decoder-only Transformer LM (RoPE, GQA, QK-norm, gated attention, SwiGLU, MoE)
world_model  action-conditioned latent world model (JEPA + SIGReg) with CEM/MPC planning
diffusion    Diffusion Transformer (adaLN-Zero) trained with flow matching / rectified flow

Every model ships three entry points: ``model.py`` (architecture), ``train.py`` and an
inference script. All of them run on a laptop CPU with the default tiny configs and scale
up by changing command-line flags.
"""

__version__ = "0.1.0"
