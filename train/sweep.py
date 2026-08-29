"""python -m train.sweep

Every condition of week 5, one row each. Resumable: conditions already present
in the output CSV are skipped, so the sweep survives an interruption.

The null baseline trains on the SAME data filtered to n_modes == 1 -- same
architecture, same optimiser, same steps, same frozen config. The only
difference is that no label set it ever sees contains a second mode.
"""
import argparse
import csv
import gc
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from sim.config import DEFAULT, Config, config_hash
from sim.generate import STRATA, split_seeds
from train.dataset import Rollouts
from train.run import split, train

FIELDS = (["cond", "k", "loss", "data", "seed", "config_hash", "success"]
          + [f"succ_{s}" for s in STRATA]
          + ["diversity", "auc_band", "auc_band_lo", "auc_band_hi",
             "auc_bar", "auc_bar_lo", "auc_bar_hi",
             "auc_pole", "auc_pole_lo", "auc_pole_hi"]
          + [f"div_{s}" for s in STRATA] + ["ckpt"])


def conditions(seeds=(0, 1, 2)):
    out = []
    for s in seeds:
        out.append(("single", 1, "rwta", "dagger", s))       # WTA == RWTA at K=1
        for k in (2, 3, 4, 6):
            for loss in ("wta", "rwta"):
                out.append((f"k{k}_{loss}", k, loss, "dagger", s))
        out.append(("null_k3", 3, "rwta", "null", s))
    return out


def unimodal_only(data: Rollouts) -> Rollouts:
    """Strictly unimodal task: every label set has exactly one mode."""
    keep = (data.n_modes == 1).nonzero().flatten()
    return data.index(keep)


def score(ckpt: str, cfg: Config, va: Rollouts, device) -> dict:
    from eval.detect import auc, auc_ci, score_dataset
    s, _, _ = score_dataset(ckpt, va, cfg, device)
    label = (va.n_modes >= 2).numpy()
    seed_id = va.seed.numpy()
    rng = va.rng.float().numpy()
    strat = va.stratum.numpy()
    lo, hi = cfg.decision_band
    band = (rng >= lo) & (rng <= hi)
    row = {}
    for tag, m in (("band", band), ("bar", band & (strat == 0)),
                   ("pole", band & (strat == 1))):
        a = auc(s[m], label[m])
        cl, ch = auc_ci(s[m], label[m], seed_id[m], n_boot=300)
        row[f"auc_{tag}"], row[f"auc_{tag}_lo"], row[f"auc_{tag}_hi"] = a, cl, ch
    return row


def diversity_by_stratum(ckpt, cfg, va, device, n=8192) -> dict:
    from train.diagnostics import by_stratum, per_sample_disagreement
    from train.student import Student
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = Student(cfg, k=blob["k"]).to(device).eval()
    model.load_state_dict(blob["state_dict"])
    b = va.index(torch.arange(min(n, len(va)))).to(device)
    with torch.no_grad():
        traj, cost = model(b.depth, b.state)
        d = by_stratum(traj, cost, b.stratum.long(), cfg.cost_margin)
        overall = float(per_sample_disagreement(traj, cost, cfg.cost_margin).mean())
    return {"diversity": overall, **{f"div_{k}": v for k, v in d.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dagger.pt")
    ap.add_argument("--out", default="results/sweep.csv")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--n-eval", type=int, default=400)
    args = ap.parse_args()

    cfg = DEFAULT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chash = config_hash(cfg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = {(r["cond"], int(r["seed"])) for r in csv.DictReader(fh)}
    else:
        with open(args.out, "w", newline="") as fh:
            csv.DictWriter(fh, FIELDS, lineterminator="\n").writeheader()

    full = Rollouts.load(args.data)
    _, va = split(full)
    va_cpu = va
    eval_seeds = split_seeds("eval", args.n_eval)
    todo = [c for c in conditions() if (c[0], c[4]) not in done]
    print(f"config {chash}   {len(todo)} conditions to run "
          f"({len(done)} already done)\n")

    from eval.harness import evaluate
    from eval.policies import student
    for cond, k, loss, which, seed in todo:
        t0 = time.perf_counter()
        data = unimodal_only(full) if which == "null" else full
        out_dir = f"runs/sweep/{cond}_s{seed}"
        _, ckpt = train(cfg, data, device, loss, args.steps, k, seed, out_dir,
                        args.steps, None, quiet=True)
        del data
        gc.collect()
        torch.cuda.empty_cache()

        rows = evaluate(lambda c: student(c, ckpt), eval_seeds, cfg, device, 400)
        succ = {f"succ_{s}": float(np.mean(
            [r["outcome"] == "success" for r in rows if r["stratum"] == s]))
            for s in STRATA}
        row = {"cond": cond, "k": k, "loss": loss, "data": which, "seed": seed,
               "config_hash": chash, "ckpt": ckpt,
               "success": float(np.mean([r["outcome"] == "success" for r in rows])),
               **succ}
        row.update(score(ckpt, cfg, va_cpu, device))
        row.update(diversity_by_stratum(ckpt, cfg, va_cpu, device))
        with open(args.out, "a", newline="") as fh:
            csv.DictWriter(fh, FIELDS, lineterminator="\n").writerow(
                {f: row.get(f, "") for f in FIELDS})
        print(f"  {cond:<10} seed {seed}  succ {row['success']:.3f}  "
              f"auc_band {row['auc_band']:.3f}  bar {row['auc_bar']:.3f}  "
              f"pole {row['auc_pole']:.3f}  div {row['diversity']:.3f}m  "
              f"[{time.perf_counter()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
