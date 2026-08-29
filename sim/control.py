"""Trajectory tracking, shared by the teacher now and the student in week 3.

It must be the SAME code in both, or the week-5 ablation compares a policy
difference and a controller difference at once.

The plan is receding-horizon: only the first segment is ever executed, and the
whole thing is replanned every step (proposal §4.1).
"""
from torch import Tensor

from .config import Config
from .dynamics import clamp_norm


def desired_velocity(traj: Tensor, cfg: Config) -> Tensor:
    """traj (N,3,3) body-frame waypoints -> (N,3) desired body velocity.

    Waypoint 0 sits at nominal time T/3 along a horizon flown at v_des, so the
    average velocity needed to reach it is waypoint / (T/3).
    """
    t1 = (cfg.horizon / cfg.v_des) / 3.0
    return clamp_norm(traj[:, 0] / t1, cfg.v_max)


def accel(traj: Tensor, vel_body: Tensor, cfg: Config) -> Tensor:
    """Velocity P-controller -> body-frame acceleration command."""
    return clamp_norm(cfg.kv * (desired_velocity(traj, cfg) - vel_body), cfg.a_max)
