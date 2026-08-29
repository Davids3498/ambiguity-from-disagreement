"""The gate tests. The first two are a PAIR: either alone passes with a broken
collision checker, and only together do they pin it."""
import torch

from eval.harness import evaluate, summarize, write_csv
from eval.policies import straight_line
from sim.config import DEFAULT as CFG
from sim.env import SUCCESS, Env
from sim.generate import BAR, POLE, generate, split_seeds
from sim.scene import blocking_box, empty_corridor
from tests.util import DEV


def test_straight_line_always_succeeds_in_an_empty_corridor():
    """No spurious collisions."""
    env = Env(CFG, DEV)
    env.reset(empty_corridor(CFG, 16, DEV))
    s = env.rollout(straight_line(CFG))
    assert bool((s["outcome"] == SUCCESS).all()), s["outcome"]


def test_straight_line_always_collides_with_a_wall_across_the_path():
    """Collisions are actually detected."""
    env = Env(CFG, DEV)
    env.reset(blocking_box(CFG, 16, DEV))
    s = env.rollout(straight_line(CFG))
    assert not bool((s["outcome"] == SUCCESS).any()), s["outcome"]


def test_straight_line_never_survives_the_ambiguous_strata():
    """If flying straight works on a bar, the bar is not an ambiguity."""
    rows = evaluate(straight_line, split_seeds("eval", 200), CFG, DEV)
    for name in ("bar", "pole"):
        sub = [r for r in rows if r["stratum"] == name]
        rate = sum(r["outcome"] == "success" for r in sub) / len(sub)
        assert rate == 0.0, f"{name}: {rate:.2f}"


def test_baselines_are_distinguishable():
    """If the two baselines score the same, the harness measures nothing."""
    from eval.policies import reactive_depth
    seeds = split_seeds("eval", 160)
    a = summarize(evaluate(straight_line, seeds, CFG, DEV))[-1]["success"]
    b = summarize(evaluate(reactive_depth, seeds, CFG, DEV))[-1]["success"]
    assert b > a + 0.15, (a, b)


def test_csv_is_byte_identical_across_runs(tmp_path):
    seeds = split_seeds("eval", 64)
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
    write_csv(evaluate(straight_line, seeds, CFG, DEV), str(p1))
    write_csv(evaluate(straight_line, seeds, CFG, DEV), str(p2))
    assert p1.read_bytes() == p2.read_bytes()


def test_batching_does_not_change_the_rows():
    seeds = split_seeds("eval", 96)
    a = evaluate(straight_line, seeds, CFG, DEV, batch=96)
    b = evaluate(straight_line, seeds, CFG, DEV, batch=7)
    assert a == b


def test_summary_rates_sum_to_one():
    rows = evaluate(straight_line, split_seeds("eval", 80), CFG, DEV)
    for r in summarize(rows):
        total = r["success"] + r["collision"] + r["timeout"] + r["oob"]
        assert abs(total - 1.0) < 1e-9, r
