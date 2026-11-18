"""python -m train.dagger --rounds 4

Round 1 the teacher flies; rounds 2+ the STUDENT flies while the teacher labels
every state it wanders into. The dataset accumulates and is
never replaced. Retraining is from scratch each round -- classical DAgger, and
at ~100 s a run there is no reason to cut the corner.

beta = 0 from round 2: the student flies alone. No perturbation is injected
during collection; that would make the week-4 recovery test much easier to pass
and is not classical DAgger.
"""
import argparse
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from sim.config import DEFAULT, Config
from sim.generate import split_seeds
from train.dataset import Rollouts, collect
from train.run import VAL_SEEDS, split, train

ROUND1_SEEDS = 2400          # week 3 used train[0:2400], of which [2000:2400] is val
PER_ROUND = 2000


def round_seeds(r: int, n: int = PER_ROUND) -> list[int]:
    """Fresh, disjoint seeds per round, all above the validation block."""
    start = ROUND1_SEEDS + (r - 2) * n
    assert start >= VAL_SEEDS[1], "round seeds must not touch validation"
    return split_seeds("train", n, start)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--data", default="data/round1.pt")
    ap.add_argument("--out", default="data/dagger.pt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--loss", choices=["wta", "rwta"], default="rwta")
    ap.add_argument("--k", type=int, default=DEFAULT.k_heads)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-ckpt", default="runs/rwta/model.pt")
    args = ap.parse_args()

    cfg = DEFAULT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Rollouts.load(args.data)
    ckpt = args.start_ckpt
    print(f"round 1 (teacher): {len(data):,} states   ckpt {ckpt}")

    from eval.policies import student
    for r in range(2, args.rounds + 1):
        t0 = time.perf_counter()
        seeds = round_seeds(r)
        prev = ckpt
        new = collect(cfg, device, seeds, batch=512, round_id=r,
                      policy_fn=lambda c, p=prev: student(c, p))
        data = Rollouts.cat([data, new])
        data.save(args.out)
        t1 = time.perf_counter()

        out_dir = f"runs/dagger_r{r}"
        train(cfg, data, device, args.loss, args.steps, args.k, args.seed,
              out_dir, args.steps, None, quiet=True)
        ckpt = f"{out_dir}/model.pt"
        tr, va = split(data)
        print(f"round {r} (student): +{len(new):,} states  "
              f"total {len(data):,} ({data.nbytes/2**30:.2f} GiB)  "
              f"train {len(tr):,} val {len(va):,}  "
              f"collect {t1-t0:.0f}s train {time.perf_counter()-t1:.0f}s -> {ckpt}")

    from eval.shift import table, tail_shift
    print(f"\n{table(data, cfg)}\n")
    print(tail_shift(data, cfg))


if __name__ == "__main__":
    main()
