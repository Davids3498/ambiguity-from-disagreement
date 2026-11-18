import math

import torch

from sim.config import DEFAULT as CFG
from sim.depth import vfov_deg
from sim.generate import (BAR, POLE, SLOT, SPLITS, STRATA, WIDE, WINDOW,
                          generate, split_seeds, subtend_px)
from tests.util import DEV


def test_splits_are_disjoint():
    for a in SPLITS:
        for b in SPLITS:
            if a == b:
                continue
            (a0, a1), (b0, b1) = SPLITS[a], SPLITS[b]
            assert a1 <= b0 or b1 <= a0, f"{a} overlaps {b}"


def test_consecutive_seeds_are_stratum_balanced():
    s = generate(CFG, split_seeds("eval", 200), DEV)
    counts = torch.bincount(s.stratum, minlength=4)
    assert (counts == 50).all(), counts


def test_seed_to_scene_is_pure_regardless_of_batch_order():
    """The same seed must give the same scene whatever else is in the batch --
    this is what a torch.Generator would not survive."""
    a = generate(CFG, [2_000_007], DEV)
    b = generate(CFG, list(range(2_000_000, 2_000_020)), DEV)
    i = 7
    assert torch.equal(a.box_lo[0], b.box_lo[i])
    assert torch.equal(a.box_hi[0], b.box_hi[i])
    assert torch.equal(a.box_on[0], b.box_on[i])
    assert torch.equal(a.cyl_c[0], b.cyl_c[i])
    assert torch.equal(a.start[0], b.start[i])


def test_bar_leaves_clearance_above_and_below():
    seeds = [s for s in split_seeds("train", 4000) if s % 4 == BAR]
    s = generate(CFG, seeds, DEV)
    k = SLOT["bar"]
    assert bool(s.box_on[:, k].all())
    below = s.box_lo[:, k, 2]
    above = CFG.height - s.box_hi[:, k, 2]
    assert below.min().item() >= CFG.bar_gap_min - 1e-4, below.min()
    assert above.min().item() >= CFG.bar_gap_min - 1e-4, above.min()


def test_pole_leaves_clearance_left_and_right():
    seeds = [s for s in split_seeds("train", 4000) if s % 4 == POLE]
    s = generate(CFG, seeds, DEV)
    y = s.cyl_c[:, 0, 1]
    assert bool(s.cyl_on[:, 0].all())
    left = CFG.width / 2 - (y + CFG.pole_r)
    right = (y - CFG.pole_r) + CFG.width / 2
    assert left.min().item() > 1.0 and right.min().item() > 1.0


def test_window_aperture_is_inside_the_corridor():
    seeds = [s for s in split_seeds("train", 4000) if s % 4 == WINDOW]
    s = generate(CFG, seeds, DEV)
    below_top = s.box_hi[:, SLOT['win_below'], 2]
    above_bot = s.box_lo[:, SLOT['win_above'], 2]
    assert (above_bot - below_top - CFG.window_ap).abs().max() < 1e-4
    assert below_top.min() > 0.3 and above_bot.max() < CFG.height - 0.3
    y_hi = s.box_lo[:, SLOT['win_py'], 1]
    y_lo = s.box_hi[:, SLOT['win_ny'], 1]
    assert (y_hi - y_lo - CFG.window_ap).abs().max() < 1e-4
    assert y_lo.min() > -CFG.width / 2 and y_hi.max() < CFG.width / 2


def test_no_distractor_inside_the_focal_band():
    s = generate(CFG, split_seeds("train", 3000), DEV)
    first = SLOT["distractor0"]
    lo = s.box_lo[:, first:, 0]
    hi = s.box_hi[:, first:, 0]
    on = s.box_on[:, first:]
    lo_edge, hi_edge = CFG.focal_x - CFG.focal_band, CFG.focal_x + CFG.focal_band
    overlaps = (hi > lo_edge) & (lo < hi_edge) & on
    assert not bool(overlaps.any()), f"{int(overlaps.sum())} distractors in band"


def test_focal_obstacles_subtend_enough_pixels():
    """At 1.54 deg/px, a 0.4 m bar at 6 m is 2.5 px and the ambiguity may
    simply not be in the image. This is the check that keeps it honest."""
    v_deg_px = vfov_deg(CFG) / CFG.cam_h
    h_deg_px = CFG.hfov_deg / CFG.cam_w
    assert subtend_px(CFG.bar_thick, CFG.decision_range, v_deg_px) >= CFG.min_subtend_px
    assert subtend_px(2 * CFG.pole_r, CFG.decision_range, h_deg_px) >= CFG.min_subtend_px
    assert subtend_px(CFG.window_ap, CFG.decision_range, v_deg_px) >= CFG.min_subtend_px


def test_start_is_always_collision_free():
    s = generate(CFG, split_seeds("train", 4000), DEV)
    d = s.distance(s.start.unsqueeze(1))[:, 0]
    assert d.min().item() > CFG.r_drone, d.min().item()


def _cross_section(s, ny=61, nz=41):
    """Free space in the corridor cross-section at x = focal_x, inflated by the
    drone radius."""
    y = torch.linspace(-CFG.width / 2, CFG.width / 2, ny, device=DEV)
    z = torch.linspace(0.0, CFG.height, nz, device=DEV)
    yy, zz = torch.meshgrid(y, z, indexing="ij")
    pts = torch.stack([torch.full_like(yy, CFG.focal_x), yy, zz], -1).reshape(-1, 3)
    free = s.distance(pts.unsqueeze(0).expand(s.n, -1, 3)) > CFG.r_drone
    return free.view(s.n, ny, nz), yy, zz


def test_focal_cross_section_has_the_intended_mode_structure():
    """The early read on the week-2 gate. This is geometry, not the
    teacher: it confirms there is physically room over AND under a bar, left AND
    right of a pole, and only one way through a window or a wide gap. It does
    NOT show the teacher will score both options as good -- that stays week 2's
    gate and must not be declared passed here.
    """
    s = generate(CFG, split_seeds("eval", 120), DEV)
    free, yy, zz = _cross_section(s)
    frac = free.float().mean(dim=(1, 2))

    # every stratum: a real obstruction, and a real way through
    assert frac.min() > 0.005 and frac.max() < 0.9, (frac.min(), frac.max())

    m = s.stratum == BAR
    zb = ((s.box_lo[m, SLOT["bar"], 2] + s.box_hi[m, SLOT["bar"], 2]) / 2)[:, None, None]
    below = (free[m] & (zz < zb)).float().mean(dim=(1, 2))
    above = (free[m] & (zz > zb)).float().mean(dim=(1, 2))
    assert below.min() > 0.02 and above.min() > 0.02, (below.min(), above.min())

    m = s.stratum == POLE
    yp = s.cyl_c[m, 0, 1][:, None, None]
    left = (free[m] & (yy > yp)).float().mean(dim=(1, 2))
    right = (free[m] & (yy < yp)).float().mean(dim=(1, 2))
    assert left.min() > 0.02 and right.min() > 0.02, (left.min(), right.min())

    # unimodal strata: ALL free space lies in the single opening
    m = s.stratum == WINDOW
    ya = ((s.box_lo[m, SLOT["win_py"], 1] + s.box_hi[m, SLOT["win_ny"], 1]) / 2)[:, None, None]
    za = ((s.box_hi[m, SLOT["win_below"], 2] + s.box_lo[m, SLOT["win_above"], 2]) / 2)[:, None, None]
    half = CFG.window_ap / 2 + 1e-3
    assert not bool((free[m] & ((yy - ya).abs() > half)).any())
    assert not bool((free[m] & ((zz - za).abs() > half)).any())

    m = s.stratum == WIDE
    yg = ((s.box_lo[m, SLOT["wide_py"], 1] + s.box_hi[m, SLOT["wide_ny"], 1]) / 2)[:, None, None]
    assert not bool((free[m] & ((yy - yg).abs() > CFG.gap_width / 2 + 1e-3)).any())


def test_focal_obstacle_intercepts_the_approach_line():
    """A bar the drone flies over without deciding is not ambiguous. The bar and
    pole strata must actually intercept the straight shot from start to goal --
    otherwise the 'canonical ambiguity' strata are mislabelled."""
    s = generate(CFG, split_seeds("eval", 400), DEV)
    t = torch.linspace(0.0, 1.0, 200, device=DEV)[None, :, None]
    line = s.start[:, None, :] + t * (s.goal - s.start)[:, None, :]
    clear = s.distance(line).amin(-1)
    blocked = clear < CFG.r_drone
    for k in (BAR, POLE):
        m = s.stratum == k
        rate = blocked[m].float().mean().item()
        assert rate > 0.98, f"{STRATA[k]}: straight line unobstructed in " \
                            f"{100*(1-rate):.0f}% of scenes"
