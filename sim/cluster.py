"""Single-linkage clustering of surviving trajectories.

Single linkage is normally criticised for chaining. Here chaining is the
required semantics: a wide gap yields a continuum of good actions differing by
one grid step at a time, and that continuum is ONE mode. A bar yields two blobs
with an infeasible band between them, and nothing bridges the gap, so they stay
TWO. Centroid methods would need K in advance -- which is the question being
asked -- and complete linkage would split a broad optimum on its diameter,
turning the wide_gap control stratum into a false positive.
"""
import math

import torch
from torch import Tensor

INF = float("inf")


def pairwise(traj: Tensor) -> Tensor:
    """(N,C,3,3) -> (N,C,C) RMS waypoint displacement, in metres.

    RMS rather than the mean of the three per-waypoint norms: it is the same
    quantity up to a constant, is still metres, is a proper metric, and is one
    cdist instead of an (N,C,C,3,3) tensor that would be 361 MB at N=1024.
    """
    n, c = traj.shape[:2]
    flat = traj.reshape(n, c, -1)
    return torch.cdist(flat, flat) / math.sqrt(traj.shape[2])


def components(dist: Tensor, active: Tensor, tau: float,
               max_iter: int = 64) -> tuple[Tensor, Tensor]:
    """Connected components by min-label propagation to a fixed point.

    Returns (label, count). Inactive entries carry the sentinel label C.
    """
    n, c, _ = dist.shape
    dev = dist.device
    idx = torch.arange(c, device=dev).expand(n, c)
    lab = torch.where(active, idx, torch.full_like(idx, c))
    adj = (dist <= tau) & active[:, :, None] & active[:, None, :]
    for _ in range(max_iter):
        prop = torch.where(adj, lab[:, None, :].expand(n, c, c),
                           torch.full((1, 1, 1), c, device=dev, dtype=lab.dtype))
        new = torch.minimum(prop.amin(-1), lab)
        if torch.equal(new, lab):
            break
        lab = new
    count = torch.zeros(n, c + 1, dtype=torch.bool, device=dev)
    count.scatter_(1, lab, torch.ones_like(lab, dtype=torch.bool))
    return lab, count[:, :c].sum(1)


def representatives(traj: Tensor, cost: Tensor, lab: Tensor, active: Tensor,
                    max_modes: int):
    """Lowest-cost member of each component, components ordered by that cost.

    Returns (traj (N,M,3,3), cost (N,M), valid (N,M)).
    """
    n, c = cost.shape
    dev = cost.device
    big = torch.where(active, cost, torch.full_like(cost, INF))

    # per-component minimum cost, then the lowest INDEX achieving it, so exact
    # ties (which symmetry makes common) resolve deterministically
    comp_cost = torch.full((n, c + 1), INF, device=dev)
    comp_cost.scatter_reduce_(1, lab, big, "amin", include_self=True)
    at_min = active & (big <= comp_cost.gather(1, lab) + 1e-12)
    idx = torch.arange(c, device=dev).expand(n, c)
    comp_idx = torch.full((n, c + 1), c, device=dev, dtype=torch.int64)
    comp_idx.scatter_reduce_(1, lab, torch.where(at_min, idx, torch.full_like(idx, c)),
                             "amin", include_self=True)
    is_rep = active & (idx == comp_idx.gather(1, lab))

    order = torch.where(is_rep, cost, torch.full_like(cost, INF)).argsort(1)
    take = order[:, :max_modes]
    return (traj.gather(1, take[:, :, None, None].expand(-1, -1, *traj.shape[2:])),
            cost.gather(1, take),
            is_rep.gather(1, take))
