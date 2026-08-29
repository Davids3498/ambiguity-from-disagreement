"""Vectorised environment. There is no single-env code path.

Finished episodes are masked, never resliced out of the batch: resizing a batch
mid-rollout is how determinism dies.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import Config
from .depth import ray_dirs, render
from .dynamics import step as dyn_step
from .scene import Scene

RUNNING, SUCCESS, COLLISION, TIMEOUT, OOB = 0, 1, 2, 3, 4
OUTCOME_NAME = {SUCCESS: "success", COLLISION: "collision",
                TIMEOUT: "timeout", OOB: "oob", RUNNING: "running"}


@dataclass
class Obs:
    """Exactly the student's week-3 input and nothing more, so there is no
    privileged leak to audit later."""
    depth: Tensor      # (N,H,W) metric metres
    vel: Tensor        # (N,3) body frame
    goal_dir: Tensor   # (N,3) body frame, unit
    goal_dist: Tensor  # (N,1)


def to_body(v: Tensor, yaw: Tensor) -> Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] + s * v[..., 1],
                        -s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


def to_world(v: Tensor, yaw: Tensor) -> Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] - s * v[..., 1],
                        s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


class Env:
    def __init__(self, cfg: Config, device):
        self.cfg, self.device = cfg, device
        self.dirs = ray_dirs(cfg.cam_w, cfg.cam_h, cfg.hfov_deg, device)

    # ---- lifecycle -------------------------------------------------------
    def reset(self, scene: Scene) -> Obs:
        n, dev = scene.n, self.device
        self.scene = scene
        self.pos = scene.start.clone()
        self.vel = torch.zeros(n, 3, device=dev)
        self.acc = torch.zeros(n, 3, device=dev)
        self.steps = torch.zeros(n, dtype=torch.int64, device=dev)
        self.outcome = torch.full((n,), RUNNING, dtype=torch.int64, device=dev)
        self.path_len = torch.zeros(n, device=dev)
        self.min_clear = torch.full((n,), float("inf"), device=dev)
        return self._observe()

    @property
    def done(self) -> Tensor:
        return self.outcome != RUNNING

    def _yaw(self) -> Tensor:
        """Camera yaw: velocity direction, falling back to the goal bearing
        below 0.1 m/s. Pitch and roll are zero -- there is no attitude."""
        vh = self.vel[:, :2]
        g = (self.scene.goal - self.pos)[:, :2]
        ref = torch.where((vh.norm(dim=-1) > 0.1).unsqueeze(-1), vh, g)
        return torch.atan2(ref[:, 1], ref[:, 0])

    def _observe(self) -> Obs:
        self.yaw = self._yaw()
        depth = render(self.scene, self.pos, self.yaw, self.dirs, self.cfg)
        gv = self.scene.goal - self.pos
        gd = gv.norm(dim=-1, keepdim=True)
        gdir = gv / gd.clamp(min=1e-6)
        return Obs(depth, to_body(self.vel, self.yaw),
                   to_body(gdir, self.yaw), gd)

    # ---- stepping --------------------------------------------------------
    def step(self, a_body: Tensor) -> Obs:
        cfg = self.cfg
        alive = ~self.done
        am = alive.unsqueeze(-1)

        a_world = to_world(a_body, self.yaw)
        pos, vel, acc = dyn_step(self.pos, self.vel, self.acc, a_world, cfg)
        self.path_len = self.path_len + torch.where(
            alive, (pos - self.pos).norm(dim=-1), torch.zeros_like(self.path_len))
        self.pos = torch.where(am, pos, self.pos)
        self.vel = torch.where(am, vel, self.vel)
        self.acc = torch.where(am, acc, self.acc)
        self.steps = self.steps + alive.to(torch.int64)

        clear = self.scene.distance(self.pos.unsqueeze(1))[:, 0]
        self.min_clear = torch.where(alive, torch.minimum(self.min_clear, clear),
                                     self.min_clear)

        p = self.pos
        oob = ((p[:, 0] < -1.0) | (p[:, 0] > cfg.length + 1.0)
               | (p[:, 1].abs() > cfg.width / 2 + 1.0)
               | (p[:, 2] < -1.0) | (p[:, 2] > cfg.height + 1.0))
        collided = clear < cfg.r_drone
        reached = p[:, 0] > cfg.length - cfg.goal_margin
        timeout = self.steps >= cfg.max_steps

        # collision beats success: touching the obstacle on the way through the
        # goal plane is a failure, not a win.
        new = torch.where(collided, COLLISION,
              torch.where(oob, OOB,
              torch.where(reached, SUCCESS,
              torch.where(timeout, TIMEOUT,
                          torch.full_like(self.outcome, RUNNING)))))
        self.outcome = torch.where(alive & (new != RUNNING), new, self.outcome)
        return self._observe()

    # ---- rollout ---------------------------------------------------------
    def rollout(self, policy, on_step=None) -> dict:
        """policy(obs, env) -- the second argument is PRIVILEGED. Only the
        teacher may read it; every student policy must ignore it."""
        obs = self._observe()
        while not bool(self.done.all()):
            a = policy(obs, self)
            if on_step is not None:
                on_step(self, obs, a)
            obs = self.step(a)
        return self.summary()

    def summary(self) -> dict:
        return {
            "outcome": self.outcome,
            "steps": self.steps,
            "time": self.steps.to(torch.float32) * self.cfg.dt,
            "path_len": self.path_len,
            "min_clear": self.min_clear,
            "mean_speed": self.path_len / (self.steps.clamp(min=1) * self.cfg.dt),
            "stratum": self.scene.stratum,
            "seed": self.scene.seeds,
        }
