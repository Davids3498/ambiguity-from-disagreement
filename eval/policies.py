"""Week-1 baselines. Both consume Obs and emit body-frame acceleration -- the
same interface the week-3 velocity P-controller will emit into, so the env never
changes when the student arrives."""
import torch
import torch.nn.functional as F

from sim.config import Config
from sim.dynamics import clamp_norm
from sim.env import Obs

def _accel(desired_v, obs: Obs, cfg: Config):
    return clamp_norm(cfg.kv * (desired_v - obs.vel), cfg.a_max)


def straight_line(cfg: Config):
    """Accelerate at the goal, ignore the depth image entirely. Establishes the
    floor and proves the harness reports failure."""
    def act(obs: Obs, env=None):
        return _accel(cfg.v_max * obs.goal_dir, obs, cfg)
    return act


def reactive_depth(cfg: Config, gain: float = 1.4, blur: int = 5,
                   brake: float = 4.0):
    """Steer toward the direction of maximum free depth, blurred, plus goal
    attraction. A non-trivial floor, and the knob for tuning obstacle density."""
    def act(obs: Obs, env=None):
        d = obs.depth.unsqueeze(1)
        k = blur
        d = F.avg_pool2d(F.pad(d, (k // 2,) * 4, mode="replicate"), k, stride=1)
        d = d.squeeze(1)                                     # (N,H,W)
        n, h, w = d.shape
        flat = d.flatten(1).argmax(1)
        row, col = flat // w, flat % w
        # pixel -> body-frame bearing through the same pinhole the sim renders
        import math
        tan_h = math.tan(math.radians(cfg.hfov_deg) / 2)
        tan_v = tan_h * h / w
        sx = (2 * (col.float() + 0.5) / w - 1) * tan_h
        sy = (2 * (row.float() + 0.5) / h - 1) * tan_v
        free = torch.stack([torch.ones_like(sx), -sx, -sy], -1)
        free = free / free.norm(dim=-1, keepdim=True)
        dir_ = free * gain + obs.goal_dir
        dir_ = dir_ / dir_.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # slow down when the path straight ahead is short
        ahead = d[:, h // 4:3 * h // 4, w // 4:3 * w // 4].amin(dim=(1, 2))
        speed = cfg.v_max * (ahead / brake).clamp(0.15, 1.0)
        return _accel(speed.unsqueeze(-1) * dir_, obs, cfg)
    return act


def teacher(cfg: Config):
    """DAgger round 1 (proposal §4.4). PRIVILEGED: reads the map and the exact
    state. Flies the lowest-cost mode, replanned every step.

    That it flies at all is the load-bearing check on the cost function -- an
    argmin that cannot complete a corridor is not one worth imitating."""
    from sim.control import accel
    from sim.teacher import label_state

    def act(obs: Obs, env):
        return accel(label_state(env, cfg).traj[:, 0], obs.vel, cfg)
    return act


def student(cfg: Config, ckpt: str):
    """The student flies the head with the LOWEST PREDICTED COST.

    That is what makes the cost head load-bearing rather than decorative: week 5
    computes disagreement only over cost-plausible heads, so a cost head that
    predicts nothing useful must show up here as a flight failure rather than as
    a surprise in the headline metric."""
    import torch
    from sim.control import accel
    from train.student import Student

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = Student(cfg, k=blob["k"]).to(blob["device"])
    model.load_state_dict(blob["state_dict"])
    model.eval()

    def act(obs: Obs, env=None):          # env is ignored: nothing privileged
        with torch.no_grad():
            st = torch.cat([obs.vel, obs.goal_dir, obs.goal_dist], -1)
            traj, cost = model(obs.depth, st)
            best = cost.argmin(1)
            pick = traj[torch.arange(traj.shape[0], device=traj.device), best]
        return accel(pick, obs.vel, cfg)
    return act


REGISTRY = {"straight_line": straight_line, "reactive_depth": reactive_depth,
            "teacher": teacher}
