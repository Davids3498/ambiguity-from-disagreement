"""Two intersection kernels and two distance functions. That is the whole
geometry layer.

Wall-with-aperture is four AABBs. The corridor shell is four AABBs. Poles are
floor-to-ceiling so cylinders need no caps. Nothing else is required.

Point-distance ships alongside ray-intersection because collision checking needs
it now and the week-2 teacher clearance term is the same function.
"""
import torch
from torch import Tensor

INF = float("inf")
_EPS = 1e-8


def _safe_inv(d: Tensor) -> Tensor:
    """1/d with the zero case pushed to a huge finite slope instead of a branch."""
    safe = torch.where(d.abs() < _EPS, torch.where(d < 0, -_EPS, _EPS), d)
    return 1.0 / safe


def ray_boxes(origin: Tensor, direction: Tensor, lo: Tensor, hi: Tensor,
              active: Tensor) -> Tensor:
    """Slab method, batched, looping over axes to keep temporaries at (N,R,B).

    origin    (N,1,3) or (N,R,3)
    direction (N,R,3), unit
    lo, hi    (N,B,3)
    active    (N,B) bool
    -> t      (N,R,B), +inf where no hit
    """
    n, r, _ = direction.shape
    b = lo.shape[1]
    tmin = direction.new_full((n, r, b), -INF)
    tmax = direction.new_full((n, r, b), INF)
    for ax in range(3):
        inv = _safe_inv(direction[..., ax].unsqueeze(-1))   # (N,R,1)
        o = origin[..., ax].unsqueeze(-1)                   # (N,R,1) or (N,1,1)
        ta = (lo[:, None, :, ax] - o) * inv                 # (N,R,B)
        tb = (hi[:, None, :, ax] - o) * inv
        tmin = torch.maximum(tmin, torch.minimum(ta, tb))
        tmax = torch.minimum(tmax, torch.maximum(ta, tb))
    t_enter = tmin.clamp(min=0.0)
    hit = (tmax >= t_enter) & (tmax >= 0.0) & active[:, None, :]
    return torch.where(hit, t_enter, torch.full_like(t_enter, INF))


def ray_cylinders(origin: Tensor, direction: Tensor, c: Tensor, r: Tensor,
                  active: Tensor) -> Tensor:
    """Vertical cylinders spanning the full corridor height: a 2-D quadratic,
    no caps.

    c (N,C,2) axis xy, r (N,C), active (N,C) -> t (N,R,C)
    """
    ox = origin[..., 0].unsqueeze(-1) - c[:, None, :, 0]    # (N,R,C)
    oy = origin[..., 1].unsqueeze(-1) - c[:, None, :, 1]
    dx = direction[..., 0].unsqueeze(-1)                    # (N,R,1)
    dy = direction[..., 1].unsqueeze(-1)
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    cc = ox * ox + oy * oy - (r[:, None, :] ** 2)
    disc = b * b - 4.0 * a * cc
    sq = torch.sqrt(disc.clamp(min=0.0))
    inv2a = 1.0 / (2.0 * a.clamp(min=1e-12))
    t0 = (-b - sq) * inv2a
    t1 = (-b + sq) * inv2a
    t = torch.where(t0 > 1e-6, t0, t1)
    ok = (disc > 0.0) & (a > 1e-12) & (t > 1e-6) & active[:, None, :]
    return torch.where(ok, t, torch.full_like(t, INF))


def dist_boxes(p: Tensor, lo: Tensor, hi: Tensor, active: Tensor) -> Tensor:
    """p (N,K,3) -> (N,K,B). Zero inside the box, exact outside."""
    pp = p.unsqueeze(2)                                     # (N,K,1,3)
    d = (lo.unsqueeze(1) - pp).clamp(min=0.0) + (pp - hi.unsqueeze(1)).clamp(min=0.0)
    return d.norm(dim=-1).masked_fill(~active[:, None, :], INF)


def dist_cylinders(p: Tensor, c: Tensor, r: Tensor, active: Tensor) -> Tensor:
    """p (N,K,3) -> (N,K,C). Infinite in z, so purely a radial distance."""
    dxy = p[..., None, :2] - c.unsqueeze(1)                 # (N,K,C,2)
    d = (dxy.norm(dim=-1) - r.unsqueeze(1)).clamp(min=0.0)
    return d.masked_fill(~active[:, None, :], INF)
