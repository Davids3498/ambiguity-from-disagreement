import math

import torch

from sim.config import DEFAULT as CFG
from sim.dynamics import clamp_norm, step
from tests.util import DEV


def _run(a_cmd, n_steps, cfg=CFG):
    p = torch.zeros(1, 3, device=DEV)
    v = torch.zeros(1, 3, device=DEV)
    a = torch.zeros(1, 3, device=DEV)
    for _ in range(n_steps):
        p, v, a = step(p, v, a, a_cmd, cfg)
    return p, v, a


def test_velocity_limit_respected():
    cmd = torch.tensor([[100.0, 0.0, 0.0]], device=DEV)
    _, v, _ = _run(cmd, 400)
    assert v.norm() <= CFG.v_max + 1e-4


def test_acceleration_limit_respected():
    cmd = torch.tensor([[1000.0, 1000.0, 1000.0]], device=DEV)
    _, _, a = _run(cmd, 200)
    assert a.norm() <= CFG.a_max + 1e-4


def test_step_response_recovers_tau():
    """After one time constant a first-order lag reaches ~63% of the command."""
    cmd = torch.tensor([[1.0, 0.0, 0.0]], device=DEV)
    n = int(round(CFG.tau / CFG.dt))
    _, _, a = _run(cmd, n)
    assert 0.55 < a[0, 0].item() < 0.72, a[0, 0].item()


def test_free_flight_matches_closed_form():
    """With no command the lag decays acc to zero and position is v*t."""
    p = torch.zeros(1, 3, device=DEV)
    v = torch.tensor([[2.0, 0.0, 0.0]], device=DEV)
    a = torch.zeros(1, 3, device=DEV)
    zero = torch.zeros(1, 3, device=DEV)
    for _ in range(50):
        p, v, a = step(p, v, a, zero, CFG)
    assert abs(p[0, 0].item() - 2.0 * 50 * CFG.dt) < 1e-4


def test_clamp_norm_preserves_direction():
    x = torch.tensor([[3.0, 4.0, 0.0]], device=DEV)
    y = clamp_norm(x, 1.0)
    assert abs(y.norm().item() - 1.0) < 1e-6
    assert torch.allclose(y * 5.0, x, atol=1e-5)


def test_clamp_norm_leaves_short_vectors_alone():
    x = torch.tensor([[0.1, 0.0, 0.0]], device=DEV)
    assert torch.allclose(clamp_norm(x, 1.0), x, atol=1e-7)
