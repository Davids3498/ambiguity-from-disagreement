"""Batched depth rendering. The chase camera in eval/record.py is this same
code at a different pose and resolution -- there is no second geometry path."""
import math

import torch
from torch import Tensor

from .config import Config
from .primitives import ray_boxes, ray_cylinders
from .scene import Scene


def vfov_deg(cfg: Config) -> float:
    tan_h = math.tan(math.radians(cfg.hfov_deg) / 2)
    return math.degrees(2 * math.atan(tan_h * cfg.cam_h / cfg.cam_w))


def ray_dirs(w: int, h: int, hfov_deg: float, device) -> Tensor:
    """(R,3) unit directions in camera frame. Row-major, row 0 = image top."""
    u = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    v = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    tan_h = math.tan(math.radians(hfov_deg) / 2)
    tan_v = tan_h * h / w
    sx = (2 * u / w - 1) * tan_h          # image-right is -y (y is left)
    sy = (2 * v / h - 1) * tan_v          # image-down  is -z
    sy_g, sx_g = torch.meshgrid(sy, sx, indexing="ij")
    d = torch.stack([torch.ones_like(sx_g), -sx_g, -sy_g], -1).reshape(-1, 3)
    return d / d.norm(dim=-1, keepdim=True)


def rotate(dirs: Tensor, yaw: Tensor, pitch: Tensor | None = None) -> Tensor:
    """(R,3) camera-frame dirs + (N,) yaw [+ (N,) pitch] -> (N,R,3) world dirs."""
    dx, dy, dz = dirs[:, 0][None], dirs[:, 1][None], dirs[:, 2][None]  # (1,R)
    if pitch is not None:
        cp, sp = torch.cos(pitch)[:, None], torch.sin(pitch)[:, None]
        dx, dz = cp * dx - sp * dz, sp * dx + cp * dz
    cy, sy = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    wx = cy * dx - sy * dy
    wy = sy * dx + cy * dy
    wz = dz.expand_as(wx)
    return torch.stack([wx, wy, wz], -1)


def trace(scene: Scene, origin: Tensor, dirs_w: Tensor, z_max: float,
          chunk: int, want_id: bool = False):
    """Running-min over primitive chunks.

    The naive (N,R,P) tensor is ~500 MB per fp32 temporary at N=1024; chunking
    holds peak at O(N*R*chunk) with no measurable slowdown, because the whole
    kernel is memory-bound. See plan_week1.md §4.1.
    """
    n, r, _ = dirs_w.shape
    best = dirs_w.new_full((n, r), z_max)
    best_id = torch.full((n, r), -1, dtype=torch.int64, device=dirs_w.device) \
        if want_id else None

    def absorb(t: Tensor, offset: int):
        nonlocal best, best_id
        if want_id:
            tv, ti = t.min(dim=2)
            upd = tv < best
            best = torch.where(upd, tv, best)
            best_id = torch.where(upd, ti + offset, best_id)
        else:
            best = torch.minimum(best, t.amin(dim=2))

    nb = scene.box_lo.shape[1]
    for i in range(0, nb, chunk):
        j = min(i + chunk, nb)
        absorb(ray_boxes(origin, dirs_w, scene.box_lo[:, i:j],
                         scene.box_hi[:, i:j], scene.box_on[:, i:j]), i)
    nc = scene.cyl_c.shape[1]
    for i in range(0, nc, chunk):
        j = min(i + chunk, nc)
        absorb(ray_cylinders(origin, dirs_w, scene.cyl_c[:, i:j],
                             scene.cyl_r[:, i:j], scene.cyl_on[:, i:j]), nb + i)
    best = best.clamp(max=z_max)
    return (best, best_id) if want_id else best


def render(scene: Scene, pos: Tensor, yaw: Tensor, dirs: Tensor,
           cfg: Config) -> Tensor:
    """-> (N, cam_h, cam_w) metric depth, clipped to z_max.

    Metric metres, not normalised: normalisation is the student's input
    transform in week 3, so the sim stays interpretable.
    """
    d_w = rotate(dirs, yaw)
    z = trace(scene, pos.unsqueeze(1), d_w, cfg.z_max, cfg.prim_chunk)
    return z.view(-1, cfg.cam_h, cfg.cam_w)
