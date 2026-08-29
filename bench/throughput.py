"""Env-steps/sec at several batch sizes with a null policy.

Not vanity: week-4 DAgger and the week-5 K sweep both multiply against this
number, and a 10x shortfall found in week 4 is a schedule failure that week 1
could have caught in an afternoon. Target: >= 5,000 env-steps/s at N=1024.
"""
import argparse
import time

import torch

from sim.config import DEFAULT, Config
from sim.env import Env
from sim.generate import generate, split_seeds


def bench(n: int, cfg: Config, device, iters: int = 60) -> tuple[float, float]:
    env = Env(cfg, device)
    env.reset(generate(cfg, split_seeds('train', n), device))
    a = torch.zeros(n, 3, device=device)
    for _ in range(10):                                   # warmup
        env.step(a)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        env.outcome.zero_()                               # keep every env alive
        env.step(a)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    return n * iters / dt, peak


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=DEFAULT.prim_chunk)
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 4096])
    args = ap.parse_args()

    from dataclasses import replace
    cfg = replace(DEFAULT, prim_chunk=args.chunk)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}  "
          f"chunk={cfg.prim_chunk}  rays={cfg.n_rays}")
    for n in args.sizes:
        sps, peak = bench(n, cfg, device)
        flag = "  <-- target 5k" if n == 1024 else ""
        print(f"  N={n:5d}   {sps:10,.0f} env-steps/s   peak {peak:7.1f} MiB{flag}")
