"""Fast correctness tests (~30 s on CPU):  python -m pytest -q

They check properties that catch real bugs, not just "it runs":
* KV-cached / chunked decoding reproduces the full forward pass
* speculative decoding with greedy sampling == plain greedy decoding of the target
* MoE routing, aux-loss-free bias balancing and active-parameter accounting
* Muon produces (approximately) orthogonal updates; schedules have the right shape
* SIGReg separates Gaussian from collapsed embeddings
* flow-matching x-prediction <-> v-prediction conversion is exact
* each model can overfit a fixed batch (loss decreases)
"""

import math

import pytest
import torch

from aistudy.common.grad import GradNoiseScale, SpikeGuard, adaptive_grad_clip_, z_loss
from aistudy.common.optim import Muon, build_optimizer, lr_schedule, zeropower_via_newtonschulz5
from aistudy.diffusion.data import render_shapes
from aistudy.diffusion.flow import FlowConfig, FlowMatching
from aistudy.diffusion.model import DiT, DiTConfig
from aistudy.llm.model import LLMConfig, Transformer
from aistudy.llm.sampling import generate, logits_to_probs, speculative_generate
from aistudy.world_model.env import EnvConfig, TwoRoomEnv, collect_random_trajectories
from aistudy.world_model.model import WMConfig, WorldModel, cem_plan, sigreg


def tiny_llm(**kw) -> Transformer:
    torch.manual_seed(0)
    cfg = dict(vocab_size=64, max_seq_len=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    cfg.update(kw)
    return Transformer(LLMConfig(**cfg))


# ----------------------------------------------------------------------------- LLM
@pytest.mark.parametrize("n_experts", [0, 4])
def test_kv_cache_matches_full_forward(n_experts):
    model = tiny_llm(n_experts=n_experts).eval()
    x = torch.randint(0, 64, (2, 40))
    with torch.no_grad():
        full = model(x)["logits"]
        caches = model.make_kv_cache(2)
        parts = [model(x[:, :16], caches=caches, start_pos=0)["logits"],      # prefill
                 model(x[:, 16:30], caches=caches, start_pos=16)["logits"]]   # chunked prefill
        parts += [model(x[:, i:i + 1], caches=caches, start_pos=i)["logits"] for i in range(30, 40)]
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-4)


def test_speculative_greedy_equals_target_greedy():
    target, draft = tiny_llm(), tiny_llm(d_model=32, n_layers=1, n_heads=2, n_kv_heads=1)
    prompt = torch.randint(0, 64, (1, 5))
    ref = generate(target, prompt, 30, temperature=0.0)
    out, stats = speculative_generate(target, draft, prompt, 30, gamma=3, temperature=0.0)
    assert torch.equal(ref, out)
    _, self_stats = speculative_generate(target, target, prompt, 30, gamma=3, temperature=0.0)
    assert self_stats["acceptance_rate"] == 1.0


def test_sampling_filters():
    logits = torch.tensor([[3.0, 2.0, 1.0, -5.0]])
    assert logits_to_probs(logits, temperature=0)[0].tolist() == [1, 0, 0, 0]
    p = logits_to_probs(logits, top_k=2)
    assert p[0, 2] == 0 and p[0, 3] == 0 and math.isclose(p.sum().item(), 1, rel_tol=1e-6)
    p = logits_to_probs(logits, min_p=0.5)
    assert p[0, 1] == 0  # e^2/e^3 = 0.37 < 0.5


def test_moe_balancing_and_param_count():
    model = tiny_llm(n_experts=8, moe_top_k=2, moe_bias_update_rate=0.05)
    x = torch.randint(0, 64, (4, 32))
    out = model(x, targets=x)
    out["loss"].backward()
    assert model.num_params(active_only=True) < model.num_params()
    moe = model.blocks[1].mlp
    load_before = moe.load.clone()
    model.update_moe_biases()
    overloaded = load_before > load_before.mean()
    assert (moe.expert_bias[overloaded] < 0).all() and moe.load.sum() == 0


def test_llm_overfits_one_batch():
    model = tiny_llm()
    opt = build_optimizer(model, "adamw", lr=3e-3, weight_decay=0.0)
    x = torch.randint(0, 64, (4, 32))
    first = None
    for _ in range(60):
        loss = model(x[:, :-1], targets=x[:, 1:])["loss"]
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


# ----------------------------------------------------------------------------- optim / grad
def test_newton_schulz_orthogonalises():
    G = torch.randn(64, 32)
    X = zeropower_via_newtonschulz5(G, steps=5)
    sv = torch.linalg.svdvals(X)
    assert sv.min() > 0.5 and sv.max() < 1.3  # NS5 lands singular values in ~[0.7, 1.2]


def test_muon_optimizer_trains():
    model = tiny_llm()
    opt = build_optimizer(model, "muon", lr=3e-3)
    assert isinstance(opt, Muon) and any(g["use_muon"] for g in opt.param_groups)
    x = torch.randint(0, 64, (4, 32))
    losses = []
    for _ in range(30):
        loss = model(x[:, :-1], targets=x[:, 1:])["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_schedules():
    assert lr_schedule(0, 100, 10) == pytest.approx(0.1)
    assert lr_schedule(9, 100, 10) == pytest.approx(1.0)
    assert lr_schedule(99, 100, 10, "cosine", min_ratio=0.1) == pytest.approx(0.1, abs=1e-3)
    assert lr_schedule(50, 100, 10, "wsd", decay_frac=0.2) == 1.0          # stable phase
    assert lr_schedule(99, 100, 10, "wsd", min_ratio=0.0, decay_frac=0.2) < 0.06


def test_spike_guard_and_noise_scale():
    guard = SpikeGuard(warmup=20, z_threshold=4.0)
    for i in range(50):
        assert not guard.should_skip(2.0 + 0.01 * math.sin(i), 1.0 + 0.01 * math.cos(i))
    assert guard.should_skip(9.0, 1.0)            # loss spike
    assert guard.should_skip(float("nan"), 1.0)   # non-finite
    # noise scale: true |G|^2 = 1, tr(Sigma) = 100  ->  B_noise = 100
    ns = GradNoiseScale(beta=0.0)
    b_s, b_b = 10, 100
    assert ns.update(1 + 100 / b_s, 1 + 100 / b_b, b_s, b_b) == pytest.approx(100.0)


def test_agc_and_zloss():
    lin = torch.nn.Linear(8, 8)
    lin.weight.grad = torch.full_like(lin.weight, 100.0)
    lin.bias.grad = torch.zeros_like(lin.bias)
    adaptive_grad_clip_(lin.parameters(), clip_factor=0.01)
    ratio = lin.weight.grad.norm(dim=1) / lin.weight.norm(dim=1).clamp_min(1e-3)
    assert (ratio <= 0.0101).all()
    assert z_loss(torch.zeros(2, 5)).item() == pytest.approx(math.log(5) ** 2, rel=1e-5)


# ----------------------------------------------------------------------------- world model
def test_env_wall_blocks_and_door_passes():
    env = TwoRoomEnv(EnvConfig())
    blocked = env.dynamics(torch.tensor([[0.44, 0.1]]), torch.tensor([[1.0, 0.0]]))
    assert blocked[0, 0] < 0.5                     # cannot enter the wall
    through = torch.tensor([[0.44, 0.5]])
    for _ in range(3):
        through = env.dynamics(through, torch.tensor([[1.0, 0.0]]))
    assert through[0, 0] > 0.5                     # walks through the door


def test_sigreg_detects_collapse():
    g = torch.Generator().manual_seed(0)
    gauss = sigreg(torch.randn(1024, 16, generator=g), generator=g)
    collapsed = sigreg(torch.zeros(1024, 16), generator=g)
    assert gauss < 0.01 < collapsed


def test_world_model_loss_and_planner_shapes():
    env = TwoRoomEnv(EnvConfig())
    data = collect_random_trajectories(env, 8, 6, torch.Generator().manual_seed(0))
    model = WorldModel(WMConfig(d_model=32, n_layers=1, n_heads=2, latent_dim=8, max_len=16))
    obs = env.render(data["states"])
    out = model.loss(obs, data["actions"], rollout_steps=3)
    out["loss"].backward()
    assert all(torch.isfinite(v) for v in out.values())
    z = model.encode(obs[:1, 0])
    plan = cem_plan(model, z, z, horizon=4, population=16, elites=4, iterations=2)
    assert plan.shape == (4, 2) and plan.abs().max() <= 1


# ----------------------------------------------------------------------------- diffusion
def test_x_prediction_conversion_is_exact():
    """If the network outputs the true x0, the implied velocity must equal eps - x0."""
    x0, eps, t = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8), torch.tensor([0.2, 0.5, 0.7, 0.9])
    x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * eps
    fm = FlowMatching(FlowConfig(prediction="x"))
    oracle = lambda x, tt, y: x0  # noqa: E731
    assert torch.allclose(fm.model_velocity(oracle, x_t, t, None), eps - x0, atol=1e-5)


def test_dit_shapes_zero_init_and_sampling():
    cfg = DiTConfig(img_size=16, patch_size=4, d_model=64, n_layers=2, n_heads=4, num_classes=4)
    model = DiT(cfg)
    x, y = torch.randn(3, 3, 16, 16), torch.tensor([0, 1, 2])
    out = model(x, torch.rand(3), y)
    assert out.shape == x.shape and out.abs().max() == 0   # adaLN-Zero + zero final layer
    fm = FlowMatching()
    imgs = fm.sample(model, (4, 3, 16, 16), torch.tensor([0, 1, 2, 3]), steps=3, cfg_scale=2.0)
    assert imgs.shape == (4, 3, 16, 16) and torch.isfinite(imgs).all()


def test_dit_overfits_tiny_dataset():
    torch.manual_seed(0)
    cfg = DiTConfig(img_size=16, patch_size=4, d_model=64, n_layers=2, n_heads=4, num_classes=4, class_dropout=0.0)
    model, fm = DiT(cfg), FlowMatching(FlowConfig(t_sampling="uniform"))
    y = torch.arange(4)
    x0 = render_shapes(y, 16, torch.Generator().manual_seed(0))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(150):
        loss = fm.loss(model, x0, y, g)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-20:]) / 20 < 0.6 * sum(losses[:20]) / 20
