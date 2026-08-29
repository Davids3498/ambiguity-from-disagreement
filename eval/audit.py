"""The week-2 gate: does the teacher's cluster count separate the ambiguous
strata from the controls?

Proposal §7 asks only that a substantial fraction of ambiguous states be
multimodal. That is necessary and not sufficient -- a teacher returning two
modes everywhere would pass it and be worthless, because the week-5 ROC would be
a diagonal. So this audit reports BOTH sides and the gate is the separation.

Stratified by RANGE, not only by obstacle type: whether a state is ambiguous
depends on how far the obstacle is, and pooling ranges averages that away.
"""
import argparse
import os

import numpy as np
import torch

from eval.policies import teacher
from sim.config import DEFAULT, Config
from sim.env import Env
from sim.generate import STRATA, generate, split_seeds
from sim.teacher import label_state, plan

RANGES = (9.0, 7.0, 6.0, 5.5, 5.0, 4.5, 4.0, 3.0, 2.0)
FIELDS = ["seed", "stratum", "range", "n_modes", "feasible"]


def approach_states(scene, cfg: Config, rng: float):
    """A state `rng` metres short of the focal plane, on the approach line,
    moving forward at the planning speed."""
    frac = (cfg.focal_x - rng - cfg.start_x) / (cfg.length - cfg.start_x)
    pos = scene.start + frac * (scene.goal - scene.start)
    d = scene.goal - scene.start
    vel = cfg.v_des * d / d.norm(dim=-1, keepdim=True)
    return pos, vel, torch.atan2(vel[:, 1], vel[:, 0])


def sweep_states(scene, cfg: Config, device, **kw) -> list[dict]:
    rows = []
    for rng in RANGES:
        pos, vel, yaw = approach_states(scene, cfg, rng)
        lab = plan(scene, pos, vel, yaw, cfg, **kw)
        for j in range(scene.n):
            rows.append({"seed": int(scene.seeds[j]),
                         "stratum": STRATA[int(scene.stratum[j])],
                         "range": float(rng),
                         "n_modes": int(lab.n_modes[j]),
                         "feasible": int(lab.feasible[j])})
    return rows


def frac_multi(rows, stratum=None, band=None) -> float:
    sub = [r for r in rows
           if (stratum is None or r["stratum"] == stratum)
           and (band is None or band[0] <= r["range"] <= band[1])]
    return float(np.mean([r["n_modes"] >= 2 for r in sub])) if sub else float("nan")


def rollout_states(cfg: Config, device, seeds, every: int = 5) -> list[dict]:
    """The distribution DAgger round 1 will actually see: states the teacher
    visits while flying, not states placed on a grid."""
    env = Env(cfg, device)
    scene = generate(cfg, seeds, device)
    obs = env.reset(scene)
    pol = teacher(cfg)
    rows, step = [], 0
    while not bool(env.done.all()):
        if step % every == 0:
            alive = ~env.done
            if bool(alive.any()):
                lab = label_state(env, cfg)
                rng = cfg.focal_x - env.pos[:, 0]
                for j in alive.nonzero().flatten().tolist():
                    rows.append({"seed": int(scene.seeds[j]),
                                 "stratum": STRATA[int(scene.stratum[j])],
                                 "range": round(float(rng[j]), 1),
                                 "n_modes": int(lab.n_modes[j]),
                                 "feasible": int(lab.feasible[j])})
        obs = env.step(pol(obs, env))
        step += 1
    return rows


def _grid(rows, title: str) -> str:
    out = [title, f"{'range':>7}" + "".join(f"{s:>11}" for s in STRATA)]
    out.append("-" * len(out[-1]))
    for rng in RANGES:
        cells = []
        for s in STRATA:
            sub = [r for r in rows if r["stratum"] == s and r["range"] == rng]
            cells.append(f"{np.mean([r['n_modes'] >= 2 for r in sub]):>11.2f}"
                         if sub else f"{'--':>11}")
        out.append(f"{rng:>6.1f}m" + "".join(cells))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = DEFAULT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = split_seeds(args.split, args.n)
    scene = generate(cfg, seeds, device)
    band = cfg.decision_band

    rows = sweep_states(scene, cfg, device)
    write_csv_rows(rows, os.path.join(args.out, f"audit_{args.split}.csv"))

    print(f"multimodality audit -- {args.split} split, n={args.n}, "
          f"tau={cfg.tau}, margin={cfg.margin_abs}+{cfg.margin_rel}*best\n")
    print(_grid(rows, "fraction of states with n_modes >= 2, by range"))
    print(f"\ndecision band {band[0]:.1f}-{band[1]:.1f}m "
          f"(derived from the {cfg.horizon:.0f}m horizon, not chosen):")
    for s in STRATA:
        print(f"    {s:<10} {frac_multi(rows, s, band):.3f}")

    print("\ntau sweep (decision band)")
    print(f"{'tau':>7}" + "".join(f"{s:>11}" for s in STRATA))
    for t in (0.5, 0.65, 0.8, 1.0, 1.3):
        r = sweep_states(scene, cfg, device, tau=t)
        print(f"{t:>7.2f}" + "".join(f"{frac_multi(r, s, band):>11.2f}" for s in STRATA))

    print("\nmargin_abs sweep (decision band)")
    print(f"{'margin':>7}" + "".join(f"{s:>11}" for s in STRATA))
    for m in (0.04, 0.08, 0.12, 0.16, 0.24):
        r = sweep_states(scene, cfg, device, margin_abs=m)
        print(f"{m:>7.2f}" + "".join(f"{frac_multi(r, s, band):>11.2f}" for s in STRATA))

    ro = rollout_states(cfg, device, seeds[:200])
    print(f"\nstates the teacher actually visits while flying (n={len(ro)}):")
    for s in STRATA:
        inband = [r for r in ro if r["stratum"] == s and band[0] <= r["range"] <= band[1]]
        allr = [r for r in ro if r["stratum"] == s]
        print(f"    {s:<10} in band {np.mean([r['n_modes'] >= 2 for r in inband]):.3f}"
              f"   whole flight {np.mean([r['n_modes'] >= 2 for r in allr]):.3f}")

    n_over = np.mean([r["n_modes"] > cfg.max_modes for r in rows])
    n_infeas = np.mean([not r["feasible"] for r in rows])
    print(f"\nn_modes > max_modes({cfg.max_modes}): {n_over:.4f}   "
          f"no feasible candidate: {n_infeas:.4f}")

    amb = min(frac_multi(rows, s, band) for s in ("bar", "pole"))
    ctl = max(frac_multi(rows, s, band) for s in ("window", "wide_gap"))
    ok = amb >= 0.60 and ctl <= 0.05
    print(f"\nGATE: ambiguous >= 0.60 -> {amb:.3f}   controls <= 0.05 -> {ctl:.3f}"
          f"   {'PASS' if ok else 'FAIL'}")


def write_csv_rows(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["seed"], r["range"])):
            w.writerow({k: (f"{r[k]:.1f}" if isinstance(r[k], float) else r[k])
                        for k in FIELDS})


if __name__ == "__main__":
    main()
