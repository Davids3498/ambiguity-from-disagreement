"""python -m train.collect --seeds 2000 --out data/round1.pt

DAgger round 1: the teacher flies and labels every state it visits.
"""
import argparse
import os
import time

import torch

from sim.config import DEFAULT
from sim.generate import split_seeds
from train.dataset import collect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--split", choices=["train", "eval"], default="train")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", default="data/round1.pt")
    ap.add_argument("--ckpt", default=None,
                    help="fly this student instead of the teacher; the teacher\n"
                         "still does the labelling")
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()

    cfg = DEFAULT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = split_seeds(args.split, args.seeds, args.offset)
    t0 = time.perf_counter()
    pol = None
    if args.ckpt:
        from eval.policies import student
        pol = lambda c: student(c, args.ckpt)
    data = collect(cfg, device, seeds, batch=args.batch, every=args.every,
                   progress=True, policy_fn=pol, round_id=args.round)
    dt = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    data.save(args.out)
    print(f"\n{data.stats(cfg)}")
    print(f"\ncollected in {dt:.1f}s ({len(data)/dt:,.0f} states/s) -> {args.out} "
          f"({os.path.getsize(args.out)/2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
