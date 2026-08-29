import torch

from eval.recover import TRIGGER_RANGE, _offsets, run
from sim.config import DEFAULT as CFG
from sim.generate import generate, split_seeds
from tests.util import DEV


def _scene(n=64):
    return generate(CFG, split_seeds("eval", n), DEV)


def test_displacement_is_a_pure_function_of_the_seed():
    a, b = _scene(), _scene()
    p = a.start.clone()
    x, dx, ok_a = _offsets(a, CFG, 1.5, p)
    y, dy, ok_b = _offsets(b, CFG, 1.5, p)
    assert torch.equal(x, y) and torch.equal(dx, dy) and torch.equal(ok_a, ok_b)


def test_no_run_ever_starts_inside_geometry():
    sc = _scene(200)
    pos = sc.start + torch.tensor([CFG.focal_x - TRIGGER_RANGE - CFG.start_x,
                                   0.0, 0.0], device=DEV)
    for mag in (1.0, 2.0, 3.0):
        new, _, ok = _offsets(sc, CFG, mag, pos)
        clear = sc.distance(new.unsqueeze(1))[:, 0]
        assert bool((clear[ok] > CFG.r_drone).all()), mag


def test_displacement_is_purely_lateral_and_vertical():
    sc = _scene()
    pos = sc.start.clone()
    new, d, ok = _offsets(sc, CFG, 1.0, pos)
    assert float((new[ok, 0] - pos[ok, 0]).abs().max()) < 1e-6
    assert torch.allclose(d.norm(dim=-1), torch.ones_like(d[:, 0]), atol=1e-5)


def test_zero_magnitude_reproduces_the_ordinary_evaluation():
    from eval.harness import evaluate
    from eval.policies import reactive_depth
    seeds = split_seeds("eval", 200)
    succ, ok, _ = run(CFG, DEV, seeds, reactive_depth, 0.0)
    rows = evaluate(reactive_depth, seeds, CFG, DEV)
    ref = sum(r["outcome"] == "success" for r in rows) / len(rows)
    assert abs(float(succ[ok].mean()) - ref) < 1e-6, (float(succ[ok].mean()), ref)


def test_larger_displacement_is_harder_for_every_policy():
    from eval.policies import reactive_depth, teacher
    seeds = split_seeds("eval", 200)
    for fn in (reactive_depth, teacher):
        prev = 1.1
        for mag in (0.0, 1.5, 2.5):
            succ, ok, _ = run(CFG, DEV, seeds, fn, mag)
            cur = float(succ[ok].mean())
            assert cur <= prev + 0.02, (fn.__name__, mag, cur, prev)
            prev = cur


def test_teacher_stays_above_the_reactive_baseline_under_displacement():
    """The ceiling must behave like a ceiling."""
    from eval.policies import reactive_depth, teacher
    seeds = split_seeds("eval", 200)
    t, ok_t, _ = run(CFG, DEV, seeds, teacher, 2.0)
    r, ok_r, _ = run(CFG, DEV, seeds, reactive_depth, 2.0)
    assert float(t[ok_t].mean()) > float(r[ok_r].mean()) + 0.15
