import math

import torch

from sim.primitives import dist_boxes, dist_cylinders, ray_boxes, ray_cylinders
from tests.util import DEV, scene_of


def test_box_face_hit_at_known_range():
    s = scene_of([([5.0, -1.0, -1.0], [6.0, 1.0, 1.0])])
    o = torch.zeros(1, 1, 3, device=DEV)
    d = torch.tensor([[[1.0, 0.0, 0.0]]], device=DEV)
    t = ray_boxes(o, d, s.box_lo, s.box_hi, s.box_on)
    assert torch.allclose(t[0, 0, 0], torch.tensor(5.0, device=DEV), atol=1e-5)


def test_box_parallel_miss_is_inf():
    s = scene_of([([5.0, -1.0, 2.0], [6.0, 1.0, 3.0])])
    o = torch.zeros(1, 1, 3, device=DEV)
    d = torch.tensor([[[1.0, 0.0, 0.0]]], device=DEV)     # passes under the box
    t = ray_boxes(o, d, s.box_lo, s.box_hi, s.box_on)
    assert torch.isinf(t).all()


def test_box_behind_camera_is_inf():
    s = scene_of([([-6.0, -1.0, -1.0], [-5.0, 1.0, 1.0])])
    o = torch.zeros(1, 1, 3, device=DEV)
    d = torch.tensor([[[1.0, 0.0, 0.0]]], device=DEV)
    t = ray_boxes(o, d, s.box_lo, s.box_hi, s.box_on)
    assert torch.isinf(t).all()


def test_inactive_primitive_never_hits():
    s = scene_of([([5.0, -1.0, -1.0], [6.0, 1.0, 1.0])])
    s.box_on[:] = False
    o = torch.zeros(1, 1, 3, device=DEV)
    d = torch.tensor([[[1.0, 0.0, 0.0]]], device=DEV)
    assert torch.isinf(ray_boxes(o, d, s.box_lo, s.box_hi, s.box_on)).all()


def test_cylinder_head_on_and_tangent():
    s = scene_of([], [(5.0, 0.0, 1.0)])
    o = torch.zeros(1, 3, 3, device=DEV)
    d = torch.tensor([[[1.0, 0.0, 0.0],           # head on -> 4.0
                       [0.0, 1.0, 0.0],           # sideways -> miss
                       [1.0, 0.0, 0.5]]], device=DEV)   # up-tilted, still hits
    d = d / d.norm(dim=-1, keepdim=True)
    t = ray_cylinders(o, d, s.cyl_c, s.cyl_r, s.cyl_on)
    assert torch.allclose(t[0, 0, 0], torch.tensor(4.0, device=DEV), atol=1e-4)
    assert torch.isinf(t[0, 1, 0])
    # cylinder is infinite in z, so the tilted ray hits at 4.0 / cos(elevation)
    expect = 4.0 / d[0, 2, 0]
    assert torch.allclose(t[0, 2, 0], expect, atol=1e-4)


def test_point_to_box_matches_dense_surface_sampling():
    lo = torch.tensor([1.0, -0.5, 0.0], device=DEV)
    hi = torch.tensor([2.0, 0.5, 1.5], device=DEV)
    s = scene_of([(lo.tolist(), hi.tolist())])
    g = torch.linspace(0, 1, 60, device=DEV)
    a, b, c = torch.meshgrid(g, g, g, indexing="ij")
    q = torch.stack([a, b, c], -1).reshape(-1, 3)
    face = ((q == 0) | (q == 1)).any(-1)
    surf = lo + (hi - lo) * q[face]                       # (S,3) surface points
    pts = torch.tensor([[[3.0, 2.0, 4.0], [1.5, 0.0, 3.0], [0.0, 0.0, 0.0]]],
                       device=DEV)
    got = dist_boxes(pts, s.box_lo, s.box_hi, s.box_on)[0, :, 0]
    ref = torch.cdist(pts[0], surf).amin(-1)
    assert torch.allclose(got, ref, atol=2e-3), (got, ref)


def test_point_inside_box_is_zero():
    s = scene_of([([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])])
    p = torch.tensor([[[0.5, 0.5, 0.5]]], device=DEV)
    assert dist_boxes(p, s.box_lo, s.box_hi, s.box_on).item() == 0.0


def test_point_to_cylinder_is_radial():
    s = scene_of([], [(0.0, 0.0, 1.0)])
    p = torch.tensor([[[3.0, 4.0, 99.0]]], device=DEV)    # infinite in z
    d = dist_cylinders(p, s.cyl_c, s.cyl_r, s.cyl_on)
    assert abs(d.item() - 4.0) < 1e-5
