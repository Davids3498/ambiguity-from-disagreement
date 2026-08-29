"""Is DAgger collecting anything new?

The premise of DAgger is that the student visits states the teacher does not. A
0.963-success student mostly flies well, so it is a live possibility that rounds
2+ collect almost nothing new -- and that has to be MEASURED before any
conclusion is drawn from the rounds.

Note on the axes: the dataset stores no world position (the student never sees
one), so "off the approach line" is measured indirectly, by how far the goal
bearing has swung out of the body-frame forward axis. The sharpest axis needs no
proxy at all: the teacher's own cost for the best trajectory it can find, which
says how bad a state is from the teacher's point of view.
"""
import numpy as np
import torch

from sim.config import Config
from train.dataset import Rollouts


def axes(d: Rollouts, cfg: Config) -> dict:
    """Label-free shift axes -- none of these read n_modes."""
    vel, gdir = d.state[:, :3], d.state[:, 3:6]
    lo, hi = cfg.decision_band
    band = (d.rng.float() >= lo) & (d.rng.float() <= hi)
    return {
        "speed": vel.norm(dim=-1),
        "goal_offaxis": gdir[:, 1:].norm(dim=-1),   # 0 = pointing at the goal
        "teacher_cost": d.cost[:, 0],               # best label's cost
        "infeasible": (d.n_modes == 0).float(),
        "_band": band,
    }


def table(data: Rollouts, cfg: Config) -> str:
    rounds = sorted(int(r) for r in data.round.unique())
    head = (f"{'round':>6}{'states':>10}{'speed':>9}{'off-axis':>10}"
            f"{'t.cost':>9}{'infeas':>9}{'band':>8}   | {'multi':>7}{'band multi':>11}")
    out = [head, "-" * len(head)]
    for r in rounds:
        d = data.index((data.round == r).nonzero().flatten())
        a = axes(d, cfg)
        b = a["_band"]
        multi = (d.n_modes >= 2).float()
        out.append(
            f"{r:>6}{len(d):>10,}{a['speed'].mean():>9.3f}"
            f"{a['goal_offaxis'].mean():>10.4f}{a['teacher_cost'].mean():>9.4f}"
            f"{a['infeasible'].mean():>9.4f}{b.float().mean():>8.4f}   |"
            f"{multi.mean():>7.4f}{multi[b].mean():>11.4f}")
    out.append("")
    out.append("  off-axis = |goal direction off the body forward axis|; "
               "t.cost = teacher cost of its own best trajectory.")
    out.append("  The two rightmost columns use the label and are reported "
               "separately: week 5 scores against them.")
    return "\n".join(out)


def tail_shift(data: Rollouts, cfg: Config, q: float = 0.99) -> str:
    """Means hide tails. DAgger's value is in the worst states, not the average
    one, so compare the upper quantile of teacher cost per round."""
    rounds = sorted(int(r) for r in data.round.unique())
    out = [f"{'round':>6}{'t.cost p50':>12}{'p90':>9}{'p99':>9}{'max':>9}"
           f"{'frac cost>0.5':>15}"]
    out.append("-" * len(out[0]))
    for r in rounds:
        c = data.cost[(data.round == r), 0].float()
        qs = torch.quantile(c, torch.tensor([0.5, 0.9, q]))
        out.append(f"{r:>6}{qs[0]:>12.4f}{qs[1]:>9.4f}{qs[2]:>9.4f}"
                   f"{c.max():>9.4f}{(c > 0.5).float().mean():>15.4f}")
    return "\n".join(out)
