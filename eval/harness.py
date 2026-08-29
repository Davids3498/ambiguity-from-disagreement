"""Seeded rollouts -> per-episode ROWS.

The harness never writes a summary number. One row per episode means week 5's
bootstrap CIs resample rows instead of re-running every condition, and the
ablation table is a groupby. Summarising at write time would force a re-run of
every experiment just to get an interval.
"""
import csv
import os

import numpy as np
import torch

from sim.config import Config
from sim.env import OUTCOME_NAME, Env
from sim.generate import STRATA, generate

FIELDS = ["seed", "stratum", "outcome", "steps", "time", "path_len",
          "min_clear", "mean_speed"]


def evaluate(policy_fn, seeds, cfg: Config, device, batch: int = 1024) -> list[dict]:
    env = Env(cfg, device)
    rows: list[dict] = []
    for i in range(0, len(seeds), batch):
        chunk = seeds[i:i + batch]
        env.reset(generate(cfg, chunk, device))
        s = env.rollout(policy_fn(cfg))
        cpu = {k: v.cpu() for k, v in s.items()}
        for j in range(len(chunk)):
            rows.append({
                "seed": int(cpu["seed"][j]),
                "stratum": STRATA[int(cpu["stratum"][j])],
                "outcome": OUTCOME_NAME[int(cpu["outcome"][j])],
                "steps": int(cpu["steps"][j]),
                "time": float(cpu["time"][j]),
                "path_len": float(cpu["path_len"][j]),
                "min_clear": float(cpu["min_clear"][j]),
                "mean_speed": float(cpu["mean_speed"][j]),
            })
    rows.sort(key=lambda r: r["seed"])
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    """Fixed precision and sorted rows, so two identical runs are byte-identical."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                        for k in FIELDS})


def read_csv(path: str) -> list[dict]:
    with open(path) as f:
        out = []
        for r in csv.DictReader(f):
            r["seed"] = int(r["seed"])
            r["steps"] = int(r["steps"])
            for k in ("time", "path_len", "min_clear", "mean_speed"):
                r[k] = float(r[k])
            out.append(r)
    return out


def _boot_ci(flags: np.ndarray, n_boot: int = 4000, seed: int = 0):
    """Percentile bootstrap over EPISODES -- the unit the rows are stored in."""
    if len(flags) == 0:
        return (float("nan"),) * 2
    rng = np.random.default_rng(seed)      # fixed: the table must reproduce
    idx = rng.integers(0, len(flags), size=(n_boot, len(flags)))
    means = flags[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict]) -> list[dict]:
    """Per-stratum, never pooled: pooled numbers hide exactly the structure this
    project is about."""
    out = []
    for name in list(STRATA) + ["ALL"]:
        sub = rows if name == "ALL" else [r for r in rows if r["stratum"] == name]
        if not sub:
            continue
        succ = np.array([r["outcome"] == "success" for r in sub], dtype=float)
        lo, hi = _boot_ci(succ)
        ok = [r for r in sub if r["outcome"] == "success"]
        out.append({
            "stratum": name,
            "n": len(sub),
            "success": succ.mean(),
            "ci_lo": lo, "ci_hi": hi,
            "collision": np.mean([r["outcome"] == "collision" for r in sub]),
            "timeout": np.mean([r["outcome"] == "timeout" for r in sub]),
            "oob": np.mean([r["outcome"] == "oob" for r in sub]),
            "t_goal": float(np.mean([r["time"] for r in ok])) if ok else float("nan"),
            "min_clear": float(np.mean([r["min_clear"] for r in sub])),
        })
    return out


def format_table(summary: list[dict]) -> str:
    head = (f"{'stratum':<10}{'n':>5}{'success':>10}{'95% CI':>16}"
            f"{'coll':>8}{'timeout':>9}{'oob':>7}{'t_goal':>9}{'clear':>8}")
    lines = [head, "-" * len(head)]
    for r in summary:
        ci = f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]"
        t = "  --  " if r["t_goal"] != r["t_goal"] else f"{r['t_goal']:.2f}s"
        sep = "-" * len(head) if r["stratum"] == "ALL" else None
        if sep:
            lines.append(sep)
        lines.append(f"{r['stratum']:<10}{r['n']:>5}{r['success']:>10.3f}{ci:>16}"
                     f"{r['collision']:>8.3f}{r['timeout']:>9.3f}{r['oob']:>7.3f}"
                     f"{t:>9}{r['min_clear']:>8.2f}")
    return "\n".join(lines)
