"""Batched scene container.

Every environment carries the SAME number of primitive slots; slots a stratum
does not use are switched off by the active mask. So there is no ragged tensor,
no padding pass, and no per-scene branching anywhere -- the mask does all of it.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import Config
from .primitives import dist_boxes, dist_cylinders


class Slots:
    """Accumulates primitive slots across the whole batch, one slot at a time."""

    def __init__(self):
        self.a, self.b, self.on = [], [], []

    def add(self, a: Tensor, b: Tensor, on: Tensor) -> None:
        self.a.append(a)
        self.b.append(b)
        self.on.append(on)

    def stack(self):
        if not self.a:
            raise ValueError("no slots added")
        return (torch.stack(self.a, 1), torch.stack(self.b, 1),
                torch.stack(self.on, 1))


@dataclass
class Scene:
    box_lo: Tensor    # (N,B,3)
    box_hi: Tensor    # (N,B,3)
    box_on: Tensor    # (N,B) bool
    cyl_c: Tensor     # (N,C,2)
    cyl_r: Tensor     # (N,C)
    cyl_on: Tensor    # (N,C) bool
    start: Tensor     # (N,3)
    goal: Tensor      # (N,3)
    stratum: Tensor   # (N,) long
    seeds: Tensor     # (N,) long

    @property
    def n(self) -> int:
        return self.box_lo.shape[0]

    def index(self, i: int) -> "Scene":
        """One-env view, for the chase camera and for debugging."""
        sl = slice(i, i + 1)
        return Scene(self.box_lo[sl], self.box_hi[sl], self.box_on[sl],
                     self.cyl_c[sl], self.cyl_r[sl], self.cyl_on[sl],
                     self.start[sl], self.goal[sl], self.stratum[sl],
                     self.seeds[sl])

    def distance(self, p: Tensor) -> Tensor:
        """Nearest-surface distance. p (N,K,3) -> (N,K). Zero inside geometry."""
        db = dist_boxes(p, self.box_lo, self.box_hi, self.box_on)
        dc = dist_cylinders(p, self.cyl_c, self.cyl_r, self.cyl_on)
        return torch.minimum(db.amin(-1), dc.amin(-1))


def shell(cfg: Config, n: int, device) -> Slots:
    """Floor, ceiling and both side walls as four thin slabs.

    They are slabs facing inward, never an inverted box, so the camera is always
    strictly outside every primitive -- which the slab kernel assumes.
    """
    s = Slots()
    t, L, W, H = 1.0, cfg.length, cfg.width, cfg.height
    on = torch.ones(n, dtype=torch.bool, device=device)

    def box(lo, hi):
        s.add(torch.tensor(lo, device=device).expand(n, 3).clone(),
              torch.tensor(hi, device=device).expand(n, 3).clone(), on.clone())

    box([-t, -W / 2 - t, -t], [L + t, W / 2 + t, 0.0])        # floor
    box([-t, -W / 2 - t, H], [L + t, W / 2 + t, H + t])       # ceiling
    box([-t, -W / 2 - t, -t], [L + t, -W / 2, H + t])         # right wall (-y)
    box([-t, W / 2, -t], [L + t, W / 2 + t, H + t])           # left wall (+y)
    return s


def pack(cfg: Config, boxes: Slots, cyls: Slots, start, goal, stratum, seeds,
          device) -> Scene:
    box_lo, box_hi, box_on = boxes.stack()
    if cyls.a:
        cyl_a, cyl_b, cyl_on = cyls.stack()
        cyl_c, cyl_r = cyl_a[..., :2], cyl_b[..., 0]
    else:
        n = box_lo.shape[0]
        cyl_c = torch.zeros(n, 1, 2, device=device)
        cyl_r = torch.zeros(n, 1, device=device)
        cyl_on = torch.zeros(n, 1, dtype=torch.bool, device=device)
    return Scene(box_lo, box_hi, box_on, cyl_c, cyl_r, cyl_on,
                 start, goal, stratum, seeds)


def _endpoints(cfg: Config, n: int, device):
    start = torch.tensor([cfg.start_x, 0.0, cfg.height / 2], device=device)
    goal = torch.tensor([cfg.length, 0.0, cfg.height / 2], device=device)
    return start.expand(n, 3).clone(), goal.expand(n, 3).clone()


def empty_corridor(cfg: Config, n: int, device) -> Scene:
    """Shell only. The false-positive half of the collision test pair."""
    start, goal = _endpoints(cfg, n, device)
    seeds = torch.zeros(n, dtype=torch.int64, device=device)
    return pack(cfg, shell(cfg, n, device), Slots(), start, goal,
                 torch.full((n,), -1, dtype=torch.int64, device=device), seeds,
                 device)


def blocking_box(cfg: Config, n: int, device) -> Scene:
    """Shell plus one slab straight across the flight path. The true-positive
    half: a straight-line policy must collide with this."""
    s = shell(cfg, n, device)
    on = torch.ones(n, dtype=torch.bool, device=device)
    lo = torch.tensor([cfg.focal_x - 0.3, -cfg.width / 2, cfg.height / 2 - 1.0],
                      device=device).expand(n, 3).clone()
    hi = torch.tensor([cfg.focal_x + 0.3, cfg.width / 2, cfg.height / 2 + 1.0],
                      device=device).expand(n, 3).clone()
    s.add(lo, hi, on)
    start, goal = _endpoints(cfg, n, device)
    seeds = torch.zeros(n, dtype=torch.int64, device=device)
    return pack(cfg, s, Slots(), start, goal,
                 torch.full((n,), -1, dtype=torch.int64, device=device), seeds,
                 device)
