"""Proposal §2 promises: 'the agent slows at a bar, stays fast through a window.'

That is an empirical claim and it has never been measured. A video can be made
to look like it is true, and a video is exactly what a reader cannot check --
so it gets measured before it gets rendered.
"""
import argparse

import numpy as np
import torch

from eval.harness import _boot_ci
from sim.config import DEFAULT
from sim.env import Env
from sim.generate import STRATA, generate, split_seeds


def speed_in_band(cfg, device, policy_fn, seeds, batch=400):
    """Mean speed inside the decision band, per stratum, one value per episode."""
    lo, hi = cfg.decision_band
    per_ep = {s: [] for s in STRATA}
    for i in range(0, len(seeds), batch):
        sc = generate(cfg, seeds[i:i + batch], device)
        env = Env(cfg, device)
        obs = env.reset(sc)
        pol = policy_fn(cfg)
        acc = torch.zeros(sc.n, device=device)
        cnt = torch.zeros(sc.n, device=device)
        while not bool(env.done.all()):
            rng = cfg.focal_x - env.pos[:, 0]
            inb = (rng >= lo) & (rng <= hi) & (~env.done)
            acc += torch.where(inb, env.vel.norm(dim=-1), torch.zeros_like(acc))
            cnt += inb.float()
            obs = env.step(pol(obs, env))
        m = cnt > 0
        for k, name in enumerate(STRATA):
            sel = m & (sc.stratum == k)
            per_ep[name] += (acc[sel] / cnt[sel]).cpu().tolist()
    return per_ep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args()
    cfg = DEFAULT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from eval.policies import student
    per_ep = speed_in_band(cfg, device, lambda c: student(c, args.ckpt),
                           split_seeds("eval", args.n))
    lo, hi = cfg.decision_band
    print(f"mean speed inside the decision band ({lo:.1f}-{hi:.1f} m), "
          f"{args.n} held-out seeds, v_max={cfg.v_max}\n")
    print(f"{'stratum':<11}{'speed m/s':>12}{'95% CI':>18}{'n':>7}")
    print("-" * 48)
    out = {}
    for name in STRATA:
        v = np.array(per_ep[name])
        c = _boot_ci(v)
        out[name] = (v.mean(), *c)
        print(f"{name:<11}{v.mean():>12.3f}   [{c[0]:.3f},{c[1]:.3f}]{len(v):>10}")
    amb = (out["bar"][0] + out["pole"][0]) / 2
    ctl = (out["window"][0] + out["wide_gap"][0]) / 2
    print(f"\n  ambiguous {amb:.3f} m/s   control {ctl:.3f} m/s   "
          f"difference {amb - ctl:+.3f} m/s")
    sep = out["bar"][2] < out["wide_gap"][1]
    print(f"  claim 'slows at a bar, stays fast through a window': "
          f"{'SUPPORTED' if sep else 'NOT SUPPORTED'} "
          f"(bar CI upper {out['bar'][2]:.3f} vs wide_gap CI lower {out['wide_gap'][1]:.3f})")


if __name__ == "__main__":
    main()
