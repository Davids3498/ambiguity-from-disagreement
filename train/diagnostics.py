"""Instrumented from training step one, not added afterwards.

Per-head win rate catches dead heads. Pairwise diversity catches duplicate
heads. NEITHER IS SUFFICIENT ALONE: uniform win rates are compatible with three
heads predicting the same trajectory, and high diversity is compatible with a
head that never wins. Both, from step one, or week 5's ablation cannot be read.
"""
import torch
import torch.nn.functional as F
from torch import Tensor

from sim.cluster import pairwise
from sim.generate import STRATA


def win_rate(winner: Tensor, valid: Tensor, k: int) -> Tensor:
    """(K,) fraction of valid labels assigned to each head."""
    oh = F.one_hot(winner, k) * valid[..., None]
    return oh.sum((0, 1)).float() / valid.sum().clamp(min=1)


def _upper(d: Tensor) -> tuple[Tensor, Tensor]:
    k = d.shape[1]
    iu = torch.triu_indices(k, k, offset=1, device=d.device)
    return iu[0], iu[1]


def diversity(pred: Tensor) -> Tensor:
    """Mean pairwise RMS between the K heads, in metres."""
    if pred.shape[1] < 2:
        return torch.zeros((), device=pred.device)
    i, j = _upper(pairwise(pred))
    return pairwise(pred)[:, i, j].mean()


def plausible_diversity(pred: Tensor, cost: Tensor, margin: float) -> Tensor:
    """Diversity over heads the model itself considers plausible -- predicted
    cost within `margin` of the best.

    This is the week-5 disagreement signal, logged from week 3 so
    it is not new, untested code at the moment it becomes the headline metric.
    A dead or garbage head must not be allowed to inflate it.
    """
    if pred.shape[1] < 2:
        return torch.zeros((), device=pred.device)
    ok = cost <= cost.amin(1, keepdim=True) + margin
    d = pairwise(pred)
    i, j = _upper(d)
    pair = (ok[:, :, None] & ok[:, None, :])[:, i, j].to(d.dtype)
    return (d[:, i, j] * pair).sum() / pair.sum().clamp(min=1.0)


def per_sample_disagreement(pred: Tensor, cost: Tensor, margin: float) -> Tensor:
    """(N,) the same quantity per state -- the raw detector score for week 5."""
    if pred.shape[1] < 2:
        return torch.zeros(pred.shape[0], device=pred.device)
    ok = cost <= cost.amin(1, keepdim=True) + margin
    d = pairwise(pred)
    i, j = _upper(d)
    pair = (ok[:, :, None] & ok[:, None, :])[:, i, j].to(d.dtype)
    return (d[:, i, j] * pair).sum(-1) / pair.sum(-1).clamp(min=1.0)


def by_stratum(pred: Tensor, cost: Tensor, stratum: Tensor, margin: float) -> dict:
    """Diversity resolved by obstacle type."""
    s = per_sample_disagreement(pred, cost, margin)
    return {name: float(s[stratum == k].mean()) if bool((stratum == k).any())
            else float("nan") for k, name in enumerate(STRATA)}
