"""Double integrator with first-order actuator lag. One substep at 20 Hz.

No attitude, no rotor dynamics: the research question is about label
multimodality and head behaviour, and physics fidelity does not move a single
reported number.
"""
import torch
from torch import Tensor

from .config import Config


def clamp_norm(x: Tensor, m: float) -> Tensor:
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return x * (n.clamp(max=m) / n)


def step(pos: Tensor, vel: Tensor, acc: Tensor, a_cmd: Tensor, cfg: Config):
    acc = acc + (cfg.dt / cfg.tau) * (a_cmd - acc)
    acc = clamp_norm(acc, cfg.a_max)
    vel = clamp_norm(vel + acc * cfg.dt, cfg.v_max)
    pos = pos + vel * cfg.dt
    return pos, vel, acc
