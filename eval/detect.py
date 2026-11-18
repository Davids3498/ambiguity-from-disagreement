"""Head disagreement as an ambiguity DETECTOR.

Predict `is_multimodal(state)` from disagreement alone, scored as a detection
problem with ROC AUC -- not as a qualitative plot. Ground truth is the teacher's
cluster count, computed at the same state with the map in hand.

Two things this module gets right on purpose:

1. Disagreement is computed only over COST-PLAUSIBLE heads (predicted cost within
   cfg.cost_margin of the best). A dead or garbage head must not inflate the
   signal -- week 3 measured a dead head reading 5.67 m of raw diversity.

2. The bootstrap resamples SEEDS, not states. States within one flight are
   correlated; resampling states would report a confidence interval several
   times narrower than the truth.
"""
import argparse

import numpy as np
import torch

from sim.config import DEFAULT, Config
from sim.generate import STRATA
from train.diagnostics import per_sample_disagreement
from train.dataset import Rollouts
from train.student import Student


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks, vectorised."""
    order = np.argsort(x, kind="mergesort")
    _, inv, counts = np.unique(x[order], return_inverse=True, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ranks = np.empty(len(x))
    ranks[order] = (starts + (counts + 1) / 2.0)[inv]
    return ranks


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney U, tie-corrected. Ties contribute 0.5, which matters here:
    a K=1 model has identically zero disagreement everywhere and must score
    exactly 0.5, not 0 or 1."""
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = _rankdata(score)
    return float((r[label].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def auc_ci(score, label, seed_id, n_boot: int = 500, rng_seed: int = 0):
    """Percentile bootstrap over SEEDS -- the independent unit."""
    rng = np.random.default_rng(rng_seed)
    seeds = np.unique(seed_id)
    idx_of = {s: np.flatnonzero(seed_id == s) for s in seeds}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        idx = np.concatenate([idx_of[s] for s in pick])
        a = auc(score[idx], label[idx])
        if a == a:
            out.append(a)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) \
        if out else (float("nan"),) * 2


@torch.no_grad()
def score_dataset(ckpt: str, data: Rollouts, cfg: Config, device,
                  batch: int = 4096):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = Student(cfg, k=blob["k"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    out = []
    for i in range(0, len(data), batch):
        b = data.index(torch.arange(i, min(i + batch, len(data)))).to(device)
        traj, cost = model(b.depth, b.state)
        out.append(per_sample_disagreement(traj, cost, cfg.cost_margin).cpu())
    return torch.cat(out).numpy(), blob["k"], blob.get("loss", "?")


def report(name: str, score, label, seed_id, mask=None) -> str:
    if mask is not None:
        score, label, seed_id = score[mask], label[mask], seed_id[mask]
    a = auc(score, label)
    n_pos = int(label.sum())
    if a != a:
        return f"  {name:<26} --      (one class absent, n={len(score):,})"
    lo, hi = auc_ci(score, label, seed_id)
    # An AUC resting on a handful of positives is not an estimate, however
    # tight the interval looks: flag it rather than let it be quoted.
    flag = "  <-- too few positives" if n_pos < 20 else ""
    return (f"  {name:<26} {a:.3f}  [{lo:.3f},{hi:.3f}]   "
            f"n={len(score):>7,}  pos={n_pos:>5,}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--data", default="data/round1.pt")
    ap.add_argument("--all", action="store_true",
                    help="score the whole file; use for a purpose-built val set")
    args = ap.parse_args()

    from train.run import split
    cfg = DEFAULT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = Rollouts.load(args.data)
    va = blob if args.all else split(blob)[1]

    label = (va.n_modes >= 2).numpy()
    seed_id = va.seed.numpy()
    rng = va.rng.float().numpy()
    strat = va.stratum.numpy()
    lo, hi = cfg.decision_band
    band = (rng >= lo) & (rng <= hi)

    print(f"validation: {len(va):,} states from {len(np.unique(seed_id))} held-out "
          f"seeds   multimodal {label.mean():.4f}   in-band {band.mean():.4f}\n")
    for ck in args.ckpt:
        score, k, loss = score_dataset(ck, va, cfg, device)
        print(f"{ck}   K={k}  loss={loss}")
        # NOTE: "all states" is dominated by states far from any obstacle,
        # where the answer is trivially "unimodal". It is reported for
        # completeness and is NOT the headline number -- the per-stratum
        # in-band rows are.
        print(report("all states (see note)", score, label, seed_id))
        print(report(f"decision band {lo:.0f}-{hi:.1f}m", score, label, seed_id, band))
        for s, nm in enumerate(STRATA):
            m = band & (strat == s)
            print(report(f"  band, {nm}", score, label, seed_id, m))
        print()


if __name__ == "__main__":
    main()
