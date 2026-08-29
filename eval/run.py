"""python -m eval.run --policy straight_line --split eval --n 200 [--record 3]"""
import argparse
import os

import torch

from eval.harness import evaluate, format_table, summarize, write_csv
from eval.policies import REGISTRY
from sim.config import DEFAULT
from sim.generate import STRATA, split_seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    help="one of %s, or 'student' with --ckpt" % sorted(REGISTRY))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", choices=["train", "eval", "test"], default="eval")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--record", type=int, default=0,
                    help="episodes per stratum to write to results/<policy>.mp4")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.split == "test":
        raise SystemExit("the test split is sealed until week 6 (plan §6)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEFAULT
    if args.policy == "student":
        if not args.ckpt:
            raise SystemExit("--policy student needs --ckpt")
        from eval.policies import student
        make = lambda c: student(c, args.ckpt)
    else:
        make = REGISTRY[args.policy]
    seeds = split_seeds(args.split, args.n, args.offset)
    rows = evaluate(make, seeds, cfg, device, args.batch)

    csv_path = os.path.join(args.out, f"{args.policy}_{args.split}.csv")
    write_csv(rows, csv_path)
    print(f"{args.policy}  {args.split} split  n={len(rows)}\n")
    print(format_table(summarize(rows)))
    print(f"\nrows -> {csv_path}")

    if args.record:
        from eval.record import record_episodes
        picks = []
        for k, name in enumerate(STRATA):
            picks += [s for s in seeds if s % len(STRATA) == k][:args.record]
        mp4 = os.path.join(args.out, f"{args.policy}_{args.split}.mp4")
        record_episodes(cfg, device, make, picks, mp4)
        print(f"video -> {mp4}  ({len(picks)} episodes)")


if __name__ == "__main__":
    main()
