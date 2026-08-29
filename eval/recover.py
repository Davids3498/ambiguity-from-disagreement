"""Recovery from states the teacher never visits (proposal §9, week-4 gate).

The teacher flies the approach line the obstacles were placed against, so an
off-axis approach at speed is BY CONSTRUCTION outside its state distribution.
This displaces the drone off that line at a fixed range and measures who gets to
the goal anyway.

Displacement is a pure function of the seed, and is checked against
`Scene.distance` so no run ever begins inside geometry -- a seed that cannot be
displaced cleanly is excluded and counted, never silently left unperturbed.
"""
import argparse
import math

import numpy as np
import torch

from sim.config import DEFAULT, Config
from sim.dynamics import clamp_norm
from sim.env import SUCCESS, Env
from sim.generate import STRATA, generate, split_seeds
from eval.harness import _boot_ci
from sim.rng import rand

TRIGGER_RANGE = 9.0     # metres short of the focal plane: time to react
HALVINGS = 3


def _offsets(scene, cfg: Config, magnitude: float, pos: torch.Tensor):
    """Per-seed direction in the body (y,z) plane, shrunk until collision-free."""
    theta = 2 * math.pi * rand(scene.seeds, 60)
    d = torch.stack([torch.zeros_like(theta), torch.cos(theta),
                     torch.sin(theta)], -1)
    scale = torch.full_like(theta, magnitude)
    ok = torch.zeros_like(theta, dtype=torch.bool)
    out = pos.clone()
    for _ in range(HALVINGS + 1):
        cand = pos + d * scale[:, None]
        clear = scene.distance(cand.unsqueeze(1))[:, 0]
        good = (clear > cfg.r_drone + 0.15) & ~ok
        out = torch.where(good[:, None], cand, out)
        ok = ok | good
        scale = torch.where(ok, scale, scale * 0.5)
    return out, d, ok


def run(cfg: Config, device, seeds, policy_fn, magnitude: float):
    env = Env(cfg, device)
    scene = generate(cfg, seeds, device)
    obs = env.reset(scene)
    pol = policy_fn(cfg)
    hit = torch.zeros(scene.n, dtype=torch.bool, device=device)
    excluded = torch.zeros(scene.n, dtype=torch.bool, device=device)
    while not bool(env.done.all()):
        trig = (~hit) & (~env.done) & (env.pos[:, 0] >= cfg.focal_x - TRIGGER_RANGE)
        if magnitude > 0 and bool(trig.any()):
            new, d, ok = _offsets(scene, cfg, magnitude, env.pos)
            m = (trig & ok).unsqueeze(-1)
            env.pos = torch.where(m, new, env.pos)
            # a matching drift, so the drone is moving off-line, not just placed
            env.vel = torch.where(m, clamp_norm(env.vel + 0.8 * magnitude * d,
                                                cfg.v_max), env.vel)
            excluded |= trig & ~ok
            hit |= trig
            obs = env._observe()
        elif bool(trig.any()):
            hit |= trig
        obs = env.step(pol(obs, env))
    ok = ~excluded
    s = env.summary()
    return ((s["outcome"] == SUCCESS) & ok).float(), ok, s["stratum"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--mag", type=float, nargs="+", default=[0.0, 1.0, 1.5, 2.0])
    ap.add_argument("--policies", nargs="+",
                    default=["reactive_depth", "teacher"])
    ap.add_argument("--ckpt", nargs="+", default=[])
    args = ap.parse_args()

    cfg, device = DEFAULT, ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = split_seeds(args.split, args.n)
    from eval.policies import REGISTRY, student

    entries = [(p, REGISTRY[p]) for p in args.policies]
    entries += [(c.split("/")[-2], (lambda cf, cc=c: student(cf, cc)))
                for c in args.ckpt]

    print(f"recovery: displaced {TRIGGER_RANGE:.0f} m short of the obstacle, "
          f"{args.n} held-out {args.split} seeds\n")
    head = f"{'policy':<18}" + "".join(f"{m:>15.1f}m" for m in args.mag)
    print(head + "\n" + "-" * len(head))
    for name, fn in entries:
        cells, excl = [], []
        for m in args.mag:
            succ, ok, _ = run(cfg, device, seeds, fn, m)
            # one episode per seed, so the bootstrap unit is the seed itself
            flags = succ[ok].cpu().numpy()
            lo, hi = _boot_ci(flags)
            cells.append(f"  {flags.mean():.3f} [{lo:.2f},{hi:.2f}]")
            excl.append(int((~ok).sum()))
        print(f"{name:<18}" + "".join(cells))
    print(f"\nseeds excluded (could not be displaced clear of geometry): "
          f"{excl}")


if __name__ == "__main__":
    main()
