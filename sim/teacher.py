"""The privileged teacher: a labelling function, not a flyer.

For any queried state it returns a LABEL SET of 1..M distinct good trajectories
(proposal §4.2). Multimodality lives WITHIN a sample, not across samples, which
is what makes the week-3 winner-takes-all assignment well-posed.

The component count this returns is simultaneously the training label and the
ground truth for the headline detection metric of proposal §2. It is the most
consequential number in the project.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from .cluster import components, pairwise, representatives
from .config import Config
from .env import to_body
from .scene import Scene

INF = float("inf")
WAYPOINT_S = (1.0 / 3.0, 2.0 / 3.0, 1.0)


@dataclass
class Labels:
    traj: Tensor      # (N,M,3,3) body-frame waypoints, cost-ordered
    cost: Tensor      # (N,M)
    valid: Tensor     # (N,M) bool -- fewer than M modes leaves trailing False
    n_modes: Tensor   # (N,) uncapped component count: the §2 ground truth
    feasible: Tensor  # (N,) whether any candidate cleared the obstacles at all


def endpoints(cfg: Config, device) -> Tensor:
    """(C,3) body-frame endpoint grid at the fixed forward horizon."""
    y = torch.linspace(-cfg.lat_max, cfg.lat_max, cfg.n_lat, device=device)
    z = torch.linspace(-cfg.vert_max, cfg.vert_max, cfg.n_vert, device=device)
    yy, zz = torch.meshgrid(y, z, indexing="ij")
    return torch.stack([torch.full_like(yy, cfg.horizon), yy, zz], -1).reshape(-1, 3)


def coeffs(v_body: Tensor, e: Tensor, cfg: Config):
    """Cubic Hermite p(s) = a1 s + a2 s^2 + a3 s^3, with p(0)=0.

    p'(0) = v_body*T holds velocity continuity -- the drone is already moving.
    p'(1) = v_des*e_hat*T means the plan is still GOING somewhere at its end,
    which a receding-horizon segment should be; a stop-at-endpoint spline would
    be a fiction the controller never executes.
    """
    t = cfg.horizon / cfg.v_des
    a1 = (v_body[:, None, :] * t).expand(-1, e.shape[0], -1)
    v1 = cfg.v_des * t * (e / e.norm(dim=-1, keepdim=True))
    return a1, 3 * e - 2 * a1 - v1, -2 * e + a1 + v1


def _eval(a1, a2, a3, s: Tensor) -> Tensor:
    s = s.view(1, 1, -1, 1)
    return (a1[:, :, None, :] * s + a2[:, :, None, :] * s ** 2
            + a3[:, :, None, :] * s ** 3)


def _curvature(a2, a3, s: Tensor) -> Tensor:
    s = s.view(1, 1, -1, 1)
    return 2 * a2[:, :, None, :] + 6 * a3[:, :, None, :] * s


def to_world(p: Tensor, pos: Tensor, yaw: Tensor) -> Tensor:
    """Body -> world for any (N, ..., 3) point tensor. Yaw only; no attitude."""
    mid = [1] * (p.dim() - 2)
    c, s = torch.cos(yaw).view(-1, *mid), torch.sin(yaw).view(-1, *mid)
    x = c * p[..., 0] - s * p[..., 1]
    y = s * p[..., 0] + c * p[..., 1]
    return torch.stack([x, y, p[..., 2]], -1) + pos.view(pos.shape[0], *mid, 3)


def plan(scene: Scene, pos: Tensor, vel: Tensor, yaw: Tensor, cfg: Config,
         tau: float | None = None, margin_abs: float | None = None,
         margin_rel: float | None = None) -> Labels:
    tau = cfg.tau if tau is None else tau
    m_abs = cfg.margin_abs if margin_abs is None else margin_abs
    m_rel = cfg.margin_rel if margin_rel is None else margin_rel

    dev = pos.device
    n = pos.shape[0]
    e = endpoints(cfg, dev)
    nc = e.shape[0]
    a1, a2, a3 = coeffs(to_body(vel, yaw), e, cfg)

    s = (torch.arange(1, cfg.n_samples + 1, device=dev, dtype=torch.float32)
         / cfg.n_samples)

    # --- clearance and feasibility, chunked over candidates ---------------
    # same (N,K,P) memory trap as the week-1 renderer, same fix.
    c_clear = torch.empty(n, nc, device=dev)
    feas = torch.empty(n, nc, dtype=torch.bool, device=dev)
    span = cfg.d_safe - cfg.r_drone
    for i in range(0, nc, cfg.cand_chunk):
        j = min(i + cfg.cand_chunk, nc)
        w = to_world(_eval(a1[:, i:j], a2[:, i:j], a3[:, i:j], s), pos, yaw)
        d = scene.distance(w.reshape(n, -1, 3)).view(n, j - i, -1)
        c_clear[:, i:j] = (((cfg.d_safe - d) / span).clamp(0, 1) ** 2).mean(-1)
        feas[:, i:j] = (d >= cfg.r_drone).all(-1)

    # --- goal progress ----------------------------------------------------
    e_world = to_world(e.expand(n, nc, 3), pos, yaw)
    d0 = (scene.goal - pos).norm(dim=-1)[:, None]
    d_end = (scene.goal[:, None, :] - e_world).norm(dim=-1)
    c_goal = (0.5 * (1.0 - (d0 - d_end) / cfg.horizon)).clamp(0, 1)

    # --- smoothness -------------------------------------------------------
    c_smooth = (_curvature(a2, a3, s).pow(2).sum(-1).mean(-1)
                / cfg.horizon ** 2)

    soft = cfg.w_clear * c_clear + cfg.w_goal * c_goal + cfg.w_smooth * c_smooth
    cost = torch.where(feas, soft, torch.full_like(soft, INF))

    # --- survivors: hybrid threshold (plan2 §3.4) -------------------------
    # A pure multiplicative factor collapses a near-zero best cost -- a wide gap
    # would come back unimodal for an arithmetic reason, not a geometric one.
    best = cost.amin(1, keepdim=True)
    surv = feas & (cost <= best + m_abs + m_rel * best)
    any_feas = feas.any(1)
    trapped = torch.zeros_like(surv)
    trapped.scatter_(1, soft.argmin(1, keepdim=True), True)
    surv = torch.where(any_feas[:, None], surv, trapped)

    # --- cluster and pick representatives ---------------------------------
    wp = _eval(a1, a2, a3, torch.tensor(WAYPOINT_S, device=dev))
    lab, n_modes = components(pairwise(wp), surv, tau)
    traj, cost_m, valid = representatives(wp, soft, lab, surv, cfg.max_modes)
    return Labels(traj, cost_m, valid,
                  torch.where(any_feas, n_modes, torch.zeros_like(n_modes)),
                  any_feas)


def label_state(env, cfg: Config, **kw) -> Labels:
    """Label the state an Env is currently in. Privileged by construction."""
    return plan(env.scene, env.pos, env.vel, env.yaw, cfg, **kw)
