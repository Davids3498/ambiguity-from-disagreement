"""Deliverables §10: ROC, K sweep, diversity by obstacle type.

matplotlib is the one dependency this project takes that does not change a
number in §6 -- it renders deliverables the proposal asks for by name.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sim.generate import STRATA

INK, GRID = "#1c1f26", "#d8dce3"
C = {"single": "#8a8f98", "wta": "#4aa8c8", "rwta": "#e88c48", "null": "#b0424a"}


def _ax(ax, xlabel, ylabel, title=None):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, color=INK, loc="left", fontweight="bold")
    ax.tick_params(colors=INK)


def roc_points(score: np.ndarray, label: np.ndarray):
    """Full ROC, tie-aware: thresholds step between distinct score values."""
    order = np.argsort(-score, kind="mergesort")
    s, y = score[order], label[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    # keep only the last index of each run of equal scores
    keep = np.r_[np.diff(s) != 0, True]
    tpr = np.r_[0.0, tp[keep] / max(tp[-1], 1)]
    fpr = np.r_[0.0, fp[keep] / max(fp[-1], 1)]
    return fpr, tpr


def roc_auc(fpr, tpr) -> float:
    return float(np.trapezoid(tpr, fpr))


def plot_roc(curves: dict, path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=160)
    _ax(ax, "false positive rate", "true positive rate", title)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls="--", zorder=1)
    for name, (fpr, tpr, col) in curves.items():
        ax.plot(fpr, tpr, color=col, lw=2.0, zorder=3,
                label=f"{name}  AUC {roc_auc(fpr, tpr):.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_k_sweep(rows: dict, path: str) -> None:
    """K against detection AUC and success rate -- proposal §2's 'K becomes
    measured, not asserted'."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=160)
    for ax, key, lab, lo in ((axes[0], "auc", "detection AUC (band, bar+pole)", 0.45),
                             (axes[1], "success", "success rate", None)):
        _ax(ax, "K (number of heads)", lab)
        for loss in ("wta", "rwta"):
            ks, mu, err = rows[loss][key]
            ax.errorbar(ks, mu, yerr=err, color=C[loss], lw=2.0, marker="o",
                        ms=5, capsize=3, label=loss.upper(), zorder=3)
        if "null" in rows and key in rows["null"]:
            # do NOT rebind ks here: the null has a single x value and would
            # clobber the axis ticks
            ax.axhline(rows["null"][key][1][0], color=C["null"], lw=1.6,
                       ls="--", zorder=2, label="null baseline")
        if lo is not None:
            ax.axhline(0.5, color=GRID, lw=1.0, ls=":", zorder=1)
            ax.set_ylim(lo, 1.02)
        ax.set_xticks(ks)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].set_title("K sweep", color=INK, loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_diversity(by_cond: dict, path: str) -> None:
    """Diversity resolved by obstacle type, against the null baseline. Without
    the null bar this plot is unfalsifiable (proposal §6)."""
    names = list(by_cond)
    x = np.arange(len(STRATA))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=160)
    _ax(ax, "", "pairwise head disagreement (m)",
        "Diversity by obstacle type, against the null baseline")
    for i, n in enumerate(names):
        mu, err = by_cond[n]
        col = C["null"] if "null" in n else (C["rwta"] if "rwta" in n else
                                             C["wta"] if "wta" in n else C["single"])
        ax.bar(x + i * w - 0.4 + w / 2, mu, w * 0.9, yerr=err, capsize=3,
               color=col, label=n, zorder=3,
               hatch="//" if "null" in n else None, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(STRATA)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
