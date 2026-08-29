import subprocess
import sys

import torch

from sim.config import DEFAULT as CFG
from sim.control import accel, desired_velocity
from sim.env import Env, to_body
from sim.generate import generate, split_seeds
from sim.teacher import (WAYPOINT_S, Labels, coeffs, endpoints, label_state,
                         plan, to_world, _eval)
from tests.util import DEV, scene_of


# ---- spline -------------------------------------------------------------
def test_spline_hits_its_endpoint_and_starts_at_the_origin():
    e = endpoints(CFG, DEV)
    v = torch.tensor([[3.0, 0.5, -0.2]], device=DEV)
    a1, a2, a3 = coeffs(v, e, CFG)
    s = torch.tensor([1e-6, 1.0], device=DEV)
    p = _eval(a1, a2, a3, s)
    assert p[0, :, 0, :].abs().max() < 1e-4                  # p(0) = 0
    assert torch.allclose(p[0, :, 1, :], e, atol=1e-4)       # p(1) = e


def test_spline_preserves_initial_velocity():
    """p'(0) = v_body * T: the drone is already moving and the plan must start
    from that, or the tracker is handed a discontinuity every step."""
    e = endpoints(CFG, DEV)
    v = torch.tensor([[3.0, 0.5, -0.2]], device=DEV)
    a1, _, _ = coeffs(v, e, CFG)
    t = CFG.horizon / CFG.v_des
    assert torch.allclose(a1[0, 0], v[0] * t, atol=1e-5)


def test_straight_endpoint_with_matched_velocity_has_no_curvature():
    e = torch.tensor([[CFG.horizon, 0.0, 0.0]], device=DEV)
    v = torch.tensor([[CFG.v_des, 0.0, 0.0]], device=DEV)
    _, a2, a3 = coeffs(v, e, CFG)
    assert a2.abs().max() < 1e-4 and a3.abs().max() < 1e-4


# ---- cost ---------------------------------------------------------------
def _symmetric_bar(z=2.5):
    """A bar exactly on the flight line, goal at the same height: over and under
    are geometrically identical."""
    return scene_of([([14.7, -4.0, z - 0.3], [15.3, 4.0, z + 0.3]),
                     ([-1.0, -4.0, -1.0], [31.0, 4.0, 0.0]),
                     ([-1.0, -4.0, 5.0], [31.0, 4.0, 6.0])],
                    start=[10.0, 0.0, z], goal=[30.0, 0.0, z])


def test_symmetric_bar_gives_equal_cost_over_and_under():
    """The single most important test in the file. If a geometrically symmetric
    obstacle does not produce two equal-cost options, no threshold choice can
    rescue the gate."""
    sc = _symmetric_bar()
    pos = torch.tensor([[10.0, 0.0, 2.5]], device=DEV)
    vel = torch.tensor([[CFG.v_des, 0.0, 0.0]], device=DEV)
    yaw = torch.zeros(1, device=DEV)
    lab = plan(sc, pos, vel, yaw, CFG)
    assert int(lab.n_modes[0]) >= 2, int(lab.n_modes[0])
    # the two cheapest modes are mirror images: equal cost, opposite z
    c = lab.cost[0, :2]
    assert abs(float(c[0] - c[1])) < 1e-4, c
    z = lab.traj[0, :2, -1, 2]
    assert float(z[0] * z[1]) < 0, z


def test_candidate_through_solid_geometry_is_infeasible():
    sc = scene_of([([12.0, -4.0, -3.0], [13.0, 4.0, 8.0])],       # full blocker
                  start=[10.0, 0.0, 2.5], goal=[30.0, 0.0, 2.5])
    pos = torch.tensor([[10.0, 0.0, 2.5]], device=DEV)
    vel = torch.tensor([[CFG.v_des, 0.0, 0.0]], device=DEV)
    lab = plan(sc, pos, vel, torch.zeros(1, device=DEV), CFG)
    assert not bool(lab.feasible[0])


def test_moving_an_obstacle_closer_never_lowers_the_clearance_cost():
    pos = torch.tensor([[10.0, 0.0, 2.5]], device=DEV)
    vel = torch.tensor([[CFG.v_des, 0.0, 0.0]], device=DEV)
    yaw = torch.zeros(1, device=DEV)
    prev = -1.0
    for dz in (3.0, 2.0, 1.5, 1.0, 0.7):
        sc = scene_of([([13.0, -4.0, 2.5 + dz], [14.0, 4.0, 2.5 + dz + 0.6])],
                      start=[10.0, 0.0, 2.5], goal=[30.0, 0.0, 2.5])
        lab = plan(sc, pos, vel, yaw, CFG)
        straight = float(lab.cost[0, 0])
        assert straight >= prev - 1e-6, (dz, straight, prev)
        prev = straight


# ---- mode structure -----------------------------------------------------
def test_bar_at_the_decision_range_is_bimodal():
    sc = _symmetric_bar()
    pos = torch.tensor([[10.0, 0.0, 2.5]], device=DEV)
    vel = torch.tensor([[CFG.v_des, 0.0, 0.0]], device=DEV)
    lab = plan(sc, pos, vel, torch.zeros(1, device=DEV), CFG)
    assert int(lab.n_modes[0]) == 2, int(lab.n_modes[0])


def test_open_corridor_is_unimodal():
    sc = scene_of([([-1.0, -4.0, -1.0], [31.0, 4.0, 0.0]),
                   ([-1.0, -4.0, 5.0], [31.0, 4.0, 6.0])],
                  start=[10.0, 0.0, 2.5], goal=[30.0, 0.0, 2.5])
    pos = torch.tensor([[10.0, 0.0, 2.5]], device=DEV)
    vel = torch.tensor([[CFG.v_des, 0.0, 0.0]], device=DEV)
    lab = plan(sc, pos, vel, torch.zeros(1, device=DEV), CFG)
    assert int(lab.n_modes[0]) == 1, int(lab.n_modes[0])


def test_label_sets_are_cost_ordered_and_valid_is_a_prefix():
    sc = generate(CFG, split_seeds("eval", 64), DEV)
    env = Env(CFG, DEV); env.reset(sc); env._observe()
    lab = label_state(env, CFG)
    c = torch.where(lab.valid, lab.cost, torch.full_like(lab.cost, float("inf")))
    assert bool((c[:, 1:] >= c[:, :-1] - 1e-6).all())
    assert bool((lab.valid.int().diff(dim=1) <= 0).all())     # no gaps
    assert bool((lab.valid.sum(1) >= 1).all())


def test_n_modes_matches_the_number_of_valid_representatives():
    sc = generate(CFG, split_seeds("eval", 128), DEV)
    env = Env(CFG, DEV); env.reset(sc); env._observe()
    lab = label_state(env, CFG)
    capped = lab.n_modes.clamp(max=CFG.max_modes)
    assert torch.equal(capped, lab.valid.sum(1))


# ---- determinism --------------------------------------------------------
HASH = """
import hashlib, torch
from sim.config import DEFAULT as CFG
from sim.env import Env
from sim.generate import generate, split_seeds
from sim.teacher import label_state
dev = "cuda" if torch.cuda.is_available() else "cpu"
env = Env(CFG, dev); env.reset(generate(CFG, split_seeds("eval", 32), dev)); env._observe()
L = label_state(env, CFG)
b = b"".join(t.cpu().numpy().tobytes() for t in (L.traj, L.cost, L.n_modes))
print(hashlib.sha256(b).hexdigest())
"""


def test_labels_identical_across_process_restarts():
    out = [subprocess.run([sys.executable, "-c", HASH], capture_output=True,
                          text=True, cwd=".", check=True).stdout.strip()
           for _ in range(2)]
    assert out[0] == out[1] and len(out[0]) == 64


def test_labels_do_not_depend_on_batch_composition():
    a = plan(*_state(generate(CFG, [2_000_005], DEV)), CFG)
    big = plan(*_state(generate(CFG, split_seeds("eval", 16), DEV)), CFG)
    assert int(a.n_modes[0]) == int(big.n_modes[5])
    assert torch.allclose(a.traj[0], big.traj[5], atol=1e-5)


def _state(sc):
    frac = (CFG.focal_x - 5.0 - CFG.start_x) / (CFG.length - CFG.start_x)
    pos = sc.start + frac * (sc.goal - sc.start)
    d = sc.goal - sc.start
    vel = CFG.v_des * d / d.norm(dim=-1, keepdim=True)
    return sc, pos, vel, torch.atan2(vel[:, 1], vel[:, 0])


# ---- control ------------------------------------------------------------
def test_controller_is_shared_between_teacher_and_a_hand_fed_trajectory():
    """Week 3's student must use this same code, or the ablation compares a
    policy difference and a controller difference at once."""
    from eval.policies import teacher
    sc = generate(CFG, split_seeds("eval", 32), DEV)
    env = Env(CFG, DEV)
    obs = env.reset(sc)
    a_pol = teacher(CFG)(obs, env)
    a_ref = accel(label_state(env, CFG).traj[:, 0], obs.vel, CFG)
    assert torch.equal(a_pol, a_ref)


def test_desired_velocity_points_at_the_first_waypoint():
    traj = torch.tensor([[[1.5, 0.5, 0.0], [3.0, 1.0, 0.0], [4.5, 1.5, 0.0]]],
                        device=DEV)
    v = desired_velocity(traj, CFG)
    assert torch.allclose(v / v.norm(), traj[:, 0] / traj[:, 0].norm(), atol=1e-5)


def test_teacher_flies_the_corridor():
    """A cost function whose argmin cannot complete a corridor is not one worth
    imitating (plan2 §6)."""
    from eval.policies import teacher
    from sim.env import SUCCESS
    from sim.generate import STRATA
    env = Env(CFG, DEV)
    env.reset(generate(CFG, split_seeds("eval", 200), DEV))
    s = env.rollout(teacher(CFG))
    ok = (s["outcome"] == SUCCESS).float()
    assert ok.mean() > 0.90, ok.mean()
    for k, name in enumerate(STRATA):
        assert ok[s["stratum"] == k].mean() > 0.90, (name, ok[s["stratum"] == k].mean())
