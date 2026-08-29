"""Teacher labelling throughput.

Week-4 DAgger relabels every state the student visits across accumulating
rounds. Target: >= 20,000 states/sec, so a 200k-state round labels in ten
seconds and the teacher never becomes the reason an experiment is slow.
"""
import argparse
import time

import torch

from sim.config import DEFAULT, Config
from sim.generate import generate, split_seeds
from sim.teacher import endpoints, plan


def bench(n: int, cfg: Config, device, iters: int = 20):
    scene = generate(cfg, split_seeds("train", n), device)
    frac = (cfg.focal_x - 5.0 - cfg.start_x) / (cfg.length - cfg.start_x)
    pos = scene.start + frac * (scene.goal - scene.start)
    d = scene.goal - scene.start
    vel = cfg.v_des * d / d.norm(dim=-1, keepdim=True)
    yaw = torch.atan2(vel[:, 1], vel[:, 0])
    for _ in range(3):
        plan(scene, pos, vel, yaw, cfg)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        plan(scene, pos, vel, yaw, cfg)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    return n * iters / dt, peak


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 4096])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEFAULT
    print(f"candidates={endpoints(cfg, device).shape[0]}  "
          f"samples={cfg.n_samples}  chunk={cfg.cand_chunk}")
    for n in args.sizes:
        sps, peak = bench(n, cfg, device)
        flag = "  <-- target 20k" if n == 1024 else ""
        print(f"  N={n:5d}   {sps:10,.0f} states/s   peak {peak:7.1f} MiB{flag}")
