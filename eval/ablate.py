"""python -m eval.ablate

Builds the week-5 tables from results/sweep.csv and refuses to build them across
runs whose frozen-config hashes disagree.

Two intervals are reported and they are NOT the same quantity:
  +-  range across the 3 training seeds  (how much the answer depends on init)
  []  percentile bootstrap over episode seeds (how much it depends on the sample)
Collapsing them into one number would overstate precision.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch

from sim.config import DEFAULT, config_hash
from sim.generate import STRATA

HEAD = ("single", "k2_wta", "k2_rwta", "k3_wta", "k3_rwta",
        "k4_wta", "k4_rwta", "k6_wta", "k6_rwta", "null_k3")


def load(path: str) -> dict:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")
    hashes = {r["config_hash"] for r in rows}
    if len(hashes) != 1:
        raise SystemExit(f"FROZEN CONFIG VIOLATED: {hashes}")
    want = config_hash(DEFAULT)
    if hashes != {want}:
        raise SystemExit(f"sweep ran under config {hashes.pop()}, current is {want}")
    by = defaultdict(list)
    for r in rows:
        by[r["cond"]].append(r)
    return by


def _agg(rs, key):
    v = np.array([float(r[key]) for r in rs])
    return v.mean(), (v.max() - v.min()) / 2.0, v


def _ci(rs, key):
    lo = np.mean([float(r[f"{key}_lo"]) for r in rs])
    hi = np.mean([float(r[f"{key}_hi"]) for r in rs])
    return lo, hi


def ablation_table(by: dict) -> str:
    h = (f"{'condition':<11}{'K':>3}{'loss':>6}{'n':>3}  {'success':>15}"
         f"{'diversity':>12}{'AUC bar':>22}{'AUC pole':>22}")
    out = [h, "-" * len(h)]
    for c in HEAD:
        if c not in by:
            continue
        rs = by[c]
        s, sr, _ = _agg(rs, "success")
        d, dr, _ = _agg(rs, "diversity")
        line = (f"{c:<11}{rs[0]['k']:>3}{rs[0]['loss']:>6}{len(rs):>3}  "
                f"{s:>8.3f} +-{sr:<4.3f}{d:>9.3f} m")
        for k in ("auc_bar", "auc_pole"):
            m, r, _ = _agg(rs, k)
            lo, hi = _ci(rs, k)
            line += f"{m:>9.3f} +-{r:<5.3f}[{lo:.2f},{hi:.2f}]"
        out.append(line)
    out += ["", "  +- = half-range over 3 training seeds;  [] = bootstrap over "
            "episode seeds (mean of the 3 runs' intervals).",
            "  window / wide_gap are absent as detection rows: 2-8 and 0 in-band "
            "positives. They are the negative class."]
    return "\n".join(out)


def k_sweep(by: dict) -> tuple[str, dict]:
    ks = [1, 2, 3, 4, 6]
    data, lines = {}, []
    h = f"{'K':>3}" + "".join(f"{l.upper():>26}" for l in ("wta", "rwta"))
    lines += [h, "-" * len(h)]
    for loss in ("wta", "rwta"):
        xs, mu, err, sc, scerr = [], [], [], [], []
        for k in ks:
            c = "single" if k == 1 else f"k{k}_{loss}"
            if c not in by:
                continue
            rs = by[c]
            a = np.array([(float(r["auc_bar"]) + float(r["auc_pole"])) / 2 for r in rs])
            s, sr, _ = _agg(rs, "success")
            xs.append(k); mu.append(a.mean()); err.append((a.max() - a.min()) / 2)
            sc.append(s); scerr.append(sr)
        data[loss] = {"auc": (xs, mu, err), "success": (xs, sc, scerr)}
    for k in ks:
        row = f"{k:>3}"
        for loss in ("wta", "rwta"):
            c = "single" if k == 1 else f"k{k}_{loss}"
            if c in by:
                rs = by[c]
                a = np.array([(float(r["auc_bar"]) + float(r["auc_pole"])) / 2
                              for r in rs])
                s, _, _ = _agg(rs, "success")
                row += f"   AUC {a.mean():.3f}+-{(a.max()-a.min())/2:.3f} succ {s:.3f}"
            else:
                row += " " * 26
        lines.append(row)
    if "null_k3" in by:
        rs = by["null_k3"]
        a = np.array([(float(r["auc_bar"]) + float(r["auc_pole"])) / 2 for r in rs])
        data["null"] = {"auc": ([3], [a.mean()], [(a.max() - a.min()) / 2])}
        lines += ["", f"  null baseline (K=3, unimodal-only data): "
                      f"AUC {a.mean():.3f} +-{(a.max()-a.min())/2:.3f}"]
    return "\n".join(lines), data


def null_gate(by: dict) -> str:
    """The scientific gate: does multimodality supervision buy anything?"""
    if "null_k3" not in by:
        return "null baseline missing"
    out = ["", "SCIENTIFIC GATE -- detection AUC vs the null baseline", "-" * 58]
    ok = True
    for key, nm in (("auc_bar", "bar"), ("auc_pole", "pole")):
        rm, rr, rv = _agg(by["k3_rwta"], key)
        nm_, nr, nv = _agg(by["null_k3"], key)
        sep = (rv.min() - nv.max()) > 0
        ok &= sep
        out.append(f"  {nm:<6} K=3 RWTA {rm:.3f} [{rv.min():.3f},{rv.max():.3f}]"
                   f"   null {nm_:.3f} [{nv.min():.3f},{nv.max():.3f}]"
                   f"   gap {rm - nm_:+.3f}   "
                   f"{'seed ranges disjoint' if sep else 'RANGES OVERLAP'}")
    out.append("")
    out.append(f"  verdict: {'PASS' if ok else 'FAIL'} -- disagreement "
               f"{'does' if ok else 'does NOT'} carry information beyond what a "
               f"unimodally-trained model has.")
    return "\n".join(out)


def diversity_table(by: dict) -> tuple[str, dict]:
    h = f"{'condition':<11}" + "".join(f"{s:>11}" for s in STRATA) + f"{'amb/ctl':>10}"
    out, plot = [h, "-" * len(h)], {}
    for c in ("single", "k3_wta", "k3_rwta", "null_k3"):
        if c not in by:
            continue
        mus, errs = [], []
        for s in STRATA:
            m, r, _ = _agg(by[c], f"div_{s}")
            mus.append(m); errs.append(r)
        amb, ctl = (mus[0] + mus[1]) / 2, (mus[2] + mus[3]) / 2
        out.append(f"{c:<11}" + "".join(f"{m:>11.3f}" for m in mus)
                   + f"{amb / max(ctl, 1e-9):>9.2f}x")
        plot[c] = (mus, errs)
    return "\n".join(out), plot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/sweep.csv")
    ap.add_argument("--data", default="data/dagger.pt")
    ap.add_argument("--out", default="results/plots")
    args = ap.parse_args()
    by = load(args.sweep)
    cfg = DEFAULT
    print(f"frozen config {config_hash(cfg)}   "
          f"{sum(len(v) for v in by.values())} runs, "
          f"{len(by)} conditions\n")
    print(ablation_table(by), "\n")
    ks, kdata = k_sweep(by)
    print(ks, "\n")
    dv, dplot = diversity_table(by)
    print(dv)
    print(null_gate(by))

    os.makedirs(args.out, exist_ok=True)
    from eval.plots import plot_diversity, plot_k_sweep, plot_roc, roc_points
    plot_k_sweep(kdata, f"{args.out}/k_sweep.png")
    plot_diversity(dplot, f"{args.out}/diversity.png")

    # ROC needs raw scores: recompute for seed-0 checkpoints of the key conditions
    from eval.detect import score_dataset
    from train.dataset import Rollouts
    from train.run import split
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, va = split(Rollouts.load(args.data))
    label = (va.n_modes >= 2).numpy()
    rng = va.rng.float().numpy()
    strat = va.stratum.numpy()
    lo, hi = cfg.decision_band
    m = (rng >= lo) & (rng <= hi) & ((strat == 0) | (strat == 1))
    curves = {}
    from eval.plots import C
    for c, col in (("single", C["single"]), ("k3_wta", C["wta"]),
                   ("k3_rwta", C["rwta"]), ("null_k3", C["null"])):
        if c not in by:
            continue
        s, _, _ = score_dataset(by[c][0]["ckpt"], va, cfg, device)
        curves[c] = (*roc_points(s[m], label[m]), col)
    plot_roc(curves, f"{args.out}/roc.png",
             "Ambiguity detection from head disagreement\ndecision band, bar + pole, held-out seeds")
    print(f"\nplots -> {args.out}/{{roc,k_sweep,diversity}}.png")


if __name__ == "__main__":
    main()
