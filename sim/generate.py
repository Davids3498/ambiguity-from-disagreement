"""Procedural scene generation. seed -> scene is a PURE function.

Constraints are satisfied by construction, never by rejection sampling: a bar's
z-centre is drawn from an interval that already guarantees >= bar_gap_min above
and below. Rejection would be a batch-ragged loop AND would make the scene
depend on draw order.

Focal obstacles are placed ON THE DRONE'S APPROACH LINE, offset by at most
cfg.focal_offset -- not at a random height or lateral position in the corridor.
This is the difference between a bar the drone must decide about and a bar it
flies over without thinking. Only the former is ambiguous, and only the former
can make the week-2 multimodality gate pass. The small random offset is what
grades the ambiguity from symmetric to lopsided, which is exactly the spread the
week-5 detection ROC needs on both sides of the decision boundary.

Four strata, chosen for their mode structure rather than for realism:
    bar      2 modes (over / under)     canonical ambiguity
    pole     2 modes (left / right)     ambiguity on a different axis
    window   1 mode                     negative control
    wide_gap 1 broad mode               a wide optimum must not read as ambiguous
"""
import torch
from torch import Tensor

from .config import Config
from .rng import rand, uniform
from .scene import Scene, Slots, pack, shell

STRATA = ("bar", "pole", "window", "wide_gap")
BAR, POLE, WINDOW, WIDE = 0, 1, 2, 3

# Every env shares this box-slot layout; unused slots are switched off, so the
# index of a primitive is a fixed, inspectable fact rather than a search.
SLOT = {"shell": 0, "bar": 4, "win_below": 5, "win_above": 6, "win_py": 7,
        "win_ny": 8, "wide_py": 9, "wide_ny": 10, "distractor0": 11}

# Explicit disjoint ranges, not `seed % k`: a modulus split is one careless
# refactor away from silent train/eval overlap. TEST is untouched until week 6.
SPLITS = {
    "train": (0, 1_000_000),
    "eval": (2_000_000, 2_010_000),
    "test": (3_000_000, 3_010_000),
}


def split_seeds(split: str, n: int, offset: int = 0) -> list[int]:
    lo, hi = SPLITS[split]
    if offset + n > hi - lo:
        raise ValueError(f"{split} split holds {hi - lo} seeds, asked for "
                         f"{offset + n}")
    return list(range(lo + offset, lo + offset + n))


def _b(*cols: Tensor) -> Tensor:
    return torch.stack(cols, -1)


def generate(cfg: Config, seeds, device) -> Scene:
    seeds = torch.as_tensor(seeds, dtype=torch.int64, device=device)
    n = seeds.shape[0]
    strat = seeds % len(STRATA)

    def const(v: float) -> Tensor:
        return torch.full((n,), float(v), device=device)

    W, H, L = cfg.width, cfg.height, cfg.length
    fx = const(cfg.focal_x)
    zero, ymin, ymax, ztop = const(0.0), const(-W / 2), const(W / 2), const(H)

    boxes = shell(cfg, n, device)
    cyls = Slots()

    # --- endpoints first: the approach line anchors every focal obstacle ----
    start = _b(const(cfg.start_x),
               uniform(seeds, 7, -1.0, 1.0),
               uniform(seeds, 8, H / 2 - 0.5, H / 2 + 0.5))
    goal = _b(const(L), zero, const(H / 2))
    frac = (cfg.focal_x - cfg.start_x) / (L - cfg.start_x)
    y_line = start[:, 1] + frac * (goal[:, 1] - start[:, 1])
    z_line = start[:, 2] + frac * (goal[:, 2] - start[:, 2])
    d = cfg.focal_offset

    # --- bar: full width, ambiguous over/under ---------------------------
    t = cfg.bar_thick
    z_lo = cfg.bar_gap_min + t / 2
    zb = (z_line + uniform(seeds, 1, -d, d)).clamp(z_lo, H - z_lo)
    boxes.add(_b(fx - t / 2, ymin, zb - t / 2),
              _b(fx + t / 2, ymax, zb + t / 2), strat == BAR)

    # --- pole: floor-to-ceiling cylinder, ambiguous left/right -----------
    yp = (y_line + uniform(seeds, 2, -d, d)).clamp(-cfg.pole_y_max, cfg.pole_y_max)
    # Slots is (a, b, on); for cylinders a[:2] is the axis xy and b[0] the radius
    cyls.add(_b(fx, yp, zero), _b(const(cfg.pole_r), zero, zero), strat == POLE)

    # --- window: wall with a square aperture, as FOUR boxes ---------------
    ap, wt = cfg.window_ap, cfg.wall_thick
    # anchored to the approach line, like every other focal obstacle. An
    # aperture 2.6 m off-axis is not a harder negative control, it is a test of
    # whether a 5 m horizon can translate 2.6 m -- a different question, and one
    # whose collision rate would contaminate every label from this stratum.
    ya = (y_line + uniform(seeds, 3, -cfg.win_off_lat, cfg.win_off_lat)) \
        .clamp(-(W / 2 - ap), W / 2 - ap)
    za = (z_line + uniform(seeds, 4, -cfg.win_off_vert, cfg.win_off_vert)) \
        .clamp(ap / 2 + 0.7, H - ap / 2 - 0.7)
    wx0, wx1 = fx - wt / 2, fx + wt / 2
    is_win = strat == WINDOW
    boxes.add(_b(wx0, ymin, zero), _b(wx1, ymax, za - ap / 2), is_win)   # below
    boxes.add(_b(wx0, ymin, za + ap / 2), _b(wx1, ymax, ztop), is_win)   # above
    boxes.add(_b(wx0, ya + ap / 2, za - ap / 2),
              _b(wx1, ymax, za + ap / 2), is_win)                        # +y
    boxes.add(_b(wx0, ymin, za - ap / 2),
              _b(wx1, ya - ap / 2, za + ap / 2), is_win)                 # -y

    # --- wide gap: one broad optimum, must not read as ambiguous ---------
    gw = cfg.gap_width
    lim = W / 2 - gw / 2 - 0.1
    yg = (y_line + uniform(seeds, 5, -0.8, 0.8)).clamp(-lim, lim)
    is_wide = strat == WIDE
    boxes.add(_b(fx - 0.3, yg + gw / 2, zero), _b(fx + 0.3, ymax, ztop), is_wide)
    boxes.add(_b(fx - 0.3, ymin, zero), _b(fx + 0.3, yg - gw / 2, ztop), is_wide)

    # --- distractors, banned from the focal band -------------------------
    # Load-bearing: a distractor beside the focal obstacle changes the true mode
    # count and quietly corrupts the week-2 ground-truth labels that the whole
    # headline metric rests on.
    n_dist = (rand(seeds, 6) * (cfg.n_distractor_max + 1)).floor()
    s = cfg.distractor_size
    near_hi = cfg.focal_x - cfg.focal_band
    far_lo = cfg.focal_x + cfg.focal_band
    for k in range(cfg.n_distractor_max):
        far = rand(seeds, 20 + 4 * k) < 0.5
        u = rand(seeds, 21 + 4 * k)
        x_near = 3.0 + u * (near_hi - 3.0 - s)
        x_far = far_lo + u * (L - 3.0 - far_lo - s)
        x = torch.where(far, x_far, x_near)
        y = uniform(seeds, 22 + 4 * k, -W / 2, W / 2 - s)
        z = uniform(seeds, 23 + 4 * k, 0.0, H - s)
        boxes.add(_b(x, y, z), _b(x + s, y + s, z + s),
                  n_dist > k)

    return pack(cfg, boxes, cyls, start, goal, strat, seeds, device)


def subtend_px(size: float, rng: float, deg_per_px: float) -> float:
    """Pixels an object of the given size spans at the given range."""
    import math
    return math.degrees(size / rng) / deg_per_px
