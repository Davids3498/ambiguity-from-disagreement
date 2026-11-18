"""Winner-takes-all and relaxed WTA assignment.

Each teacher label in the set is assigned to its nearest head; only that head
receives gradient. A head may win several labels; a label is won
by exactly one head.

RWTA gives the winner 1-relax and splits relax among the losers, which keeps
dead heads alive. It does NOT prevent duplicate heads -- that needs a separate
metric, which is why train/diagnostics.py logs two things and not one.
"""
import torch
import torch.nn.functional as F
from torch import Tensor

from sim.config import Config


def rms(a: Tensor, b: Tensor) -> Tensor:
    """RMS displacement over the 3 waypoints, in metres. Same metric as
    sim.cluster.pairwise, so head distance and cluster distance are comparable."""
    return (a - b).pow(2).sum(-1).mean(-1).sqrt()


def assign(pred: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """pred (N,K,3,3), labels (N,M,3,3) -> winner (N,M), dist (N,M,K)."""
    d = rms(pred[:, None], labels[:, :, None])
    return d.argmin(-1), d


def weights(winner: Tensor, k: int, relax: float) -> Tensor:
    """(N,M) winner index -> (N,M,K) per-head weights summing to 1."""
    w = F.one_hot(winner, k).to(torch.float32)
    if relax > 0.0 and k > 1:
        w = (1.0 - relax) * w + (relax / (k - 1)) * (1.0 - w)
    return w


def losses(pred: Tensor, pred_cost: Tensor, labels: Tensor, label_cost: Tensor,
           valid: Tensor, cfg: Config, relax: float):
    """-> (waypoint loss, cost loss, winner index). Both losses are means over
    VALID labels, so a state with two modes contributes twice."""
    k = pred.shape[1]
    winner, _ = assign(pred, labels)
    w = weights(winner, k, relax)
    sq = (pred[:, None] - labels[:, :, None]).pow(2).sum(-1).mean(-1)   # m^2
    l_wp = (w * sq).sum(-1) / cfg.horizon ** 2
    l_c = (w * (pred_cost[:, None] - label_cost[:, :, None]).pow(2)).sum(-1)
    m = valid.to(l_wp.dtype)
    den = m.sum().clamp(min=1.0)
    return (l_wp * m).sum() / den, (l_c * m).sum() / den, winner


def total(wp: Tensor, cost: Tensor, cfg: Config) -> Tensor:
    return wp + cfg.w_cost * cost
