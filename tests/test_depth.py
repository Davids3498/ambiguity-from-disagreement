import math

import torch

from sim.config import DEFAULT as CFG
from sim.depth import ray_dirs, render, rotate, vfov_deg
from tests.util import DEV, scene_of


def _brute(origin, dirs, boxes, cyls, z_max):
    """Deliberately naive, independent reimplementation. Python loops, no torch."""
    out = []
    for d in dirs:
        best = z_max
        for lo, hi in boxes:
            tmin, tmax = -math.inf, math.inf
            for ax in range(3):
                if abs(d[ax]) < 1e-12:
                    if origin[ax] < lo[ax] or origin[ax] > hi[ax]:
                        tmin, tmax = math.inf, -math.inf
                else:
                    t1 = (lo[ax] - origin[ax]) / d[ax]
                    t2 = (hi[ax] - origin[ax]) / d[ax]
                    tmin = max(tmin, min(t1, t2))
                    tmax = min(tmax, max(t1, t2))
            if tmax >= max(tmin, 0.0) and tmax >= 0.0:
                best = min(best, max(tmin, 0.0))
        for cx, cy, r in cyls:
            ox, oy = origin[0] - cx, origin[1] - cy
            a = d[0] ** 2 + d[1] ** 2
            if a > 1e-12:
                b = 2 * (ox * d[0] + oy * d[1])
                c = ox * ox + oy * oy - r * r
                disc = b * b - 4 * a * c
                if disc > 0:
                    sq = math.sqrt(disc)
                    t0, t1 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
                    t = t0 if t0 > 1e-6 else t1
                    if t > 1e-6:
                        best = min(best, t)
        out.append(best)
    return out


def test_vectorised_matches_brute_force():
    boxes = [([4.0, -2.0, -1.0], [5.0, 0.0, 1.0]),
             ([7.0, -3.0, -3.0], [9.0, 3.0, 0.2]),
             ([2.0, 1.0, -1.0], [2.5, 3.0, 2.0])]
    cyls = [(6.0, 1.2, 0.5)]
    s = scene_of(boxes, cyls)
    dirs = ray_dirs(CFG.cam_w, CFG.cam_h, CFG.hfov_deg, DEV)
    pos = torch.tensor([[0.0, 0.0, 0.0]], device=DEV)
    yaw = torch.zeros(1, device=DEV)
    got = render(s, pos, yaw, dirs, CFG).reshape(-1)
    ref = _brute([0.0, 0.0, 0.0], dirs.tolist(), boxes, cyls, CFG.z_max)
    ref = torch.tensor(ref, device=DEV, dtype=torch.float32)
    assert torch.allclose(got, ref, atol=1e-3), (got - ref).abs().max()


def test_empty_scene_is_all_z_max():
    s = scene_of([])
    dirs = ray_dirs(CFG.cam_w, CFG.cam_h, CFG.hfov_deg, DEV)
    d = render(s, torch.zeros(1, 3, device=DEV), torch.zeros(1, device=DEV),
               dirs, CFG)
    assert (d == CFG.z_max).all()


def test_depth_is_never_negative_and_is_clipped():
    boxes = [([3.0, -5.0, -5.0], [4.0, 5.0, 5.0])]
    s = scene_of(boxes)
    dirs = ray_dirs(CFG.cam_w, CFG.cam_h, CFG.hfov_deg, DEV)
    d = render(s, torch.zeros(1, 3, device=DEV), torch.zeros(1, device=DEV),
               dirs, CFG)
    assert (d >= 0).all() and (d <= CFG.z_max).all()


def test_chunk_size_does_not_change_the_image():
    from dataclasses import replace
    boxes = [([4.0 + i, -1.0, -1.0], [4.5 + i, 1.0, 1.0]) for i in range(11)]
    s = scene_of(boxes)
    dirs = ray_dirs(CFG.cam_w, CFG.cam_h, CFG.hfov_deg, DEV)
    p, y = torch.zeros(1, 3, device=DEV), torch.zeros(1, device=DEV)
    a = render(s, p, y, dirs, replace(CFG, prim_chunk=1))
    b = render(s, p, y, dirs, replace(CFG, prim_chunk=8))
    c = render(s, p, y, dirs, replace(CFG, prim_chunk=64))
    assert torch.equal(a, b) and torch.equal(b, c)


def test_yaw_rotation_is_consistent_with_moving_the_obstacle():
    """Rendering a box to the left with yaw=0 equals rendering a box straight
    ahead with yaw=+90deg."""
    ahead = scene_of([([4.0, -1.0, -1.0], [5.0, 1.0, 1.0])])
    left = scene_of([([-1.0, 4.0, -1.0], [1.0, 5.0, 1.0])])
    dirs = ray_dirs(CFG.cam_w, CFG.cam_h, CFG.hfov_deg, DEV)
    p = torch.zeros(1, 3, device=DEV)
    a = render(ahead, p, torch.zeros(1, device=DEV), dirs, CFG)
    b = render(left, p, torch.full((1,), math.pi / 2, device=DEV), dirs, CFG)
    assert torch.allclose(a, b, atol=1e-4)


def test_pitch_rotation_lifts_the_horizon():
    s = scene_of([([-20.0, -20.0, -1.0], [20.0, 20.0, 0.0])])   # ground plane
    dirs = ray_dirs(64, 48, CFG.hfov_deg, DEV)
    p = torch.tensor([[0.0, 0.0, 2.0]], device=DEV)
    from sim.depth import trace
    flat = trace(s, p.unsqueeze(1), rotate(dirs, torch.zeros(1, device=DEV)),
                 CFG.z_max, 8)
    down = trace(s, p.unsqueeze(1),
                 rotate(dirs, torch.zeros(1, device=DEV),
                        torch.full((1,), -0.3, device=DEV)), CFG.z_max, 8)
    assert (down < CFG.z_max).sum() > (flat < CFG.z_max).sum()


def test_angular_resolution_matches_plan_arithmetic():
    """plan_week1.md §4.3 claims 1.54 deg/px vertical at 64x48, hfov 90."""
    assert abs(vfov_deg(CFG) - 73.74) < 0.05
    assert abs(vfov_deg(CFG) / CFG.cam_h - 1.536) < 0.005
