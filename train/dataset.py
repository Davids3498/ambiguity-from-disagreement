"""Round-1 dataset: every state the teacher visits, with its label set.

The dataset ACCUMULATES and is never replaced (proposal §4.4) -- week 4 appends
DAgger rounds to it, so this on-disk format is fixed now.

`n_modes` is stored because week 5 scores against it. It is never sampled on:
conditioning the training distribution on the teacher's answer would make the
headline ROC a foregone conclusion rather than a finding.
"""
from dataclasses import asdict, dataclass, fields

import torch
from torch import Tensor

from eval.policies import teacher
from sim.config import Config
from sim.env import Env
from sim.generate import generate
from sim.teacher import label_state


@dataclass
class Rollouts:
    depth: Tensor     # (N,48,64) float16 -- error far below one depth pixel
    state: Tensor     # (N,7) vel(3) goal_dir(3) goal_dist(1), body frame
    traj: Tensor      # (N,M,3,3) teacher label set, body frame
    cost: Tensor      # (N,M)
    valid: Tensor     # (N,M) bool
    n_modes: Tensor   # (N,) int8   -- ground truth; never a sampling key
    stratum: Tensor   # (N,) int8
    rng: Tensor       # (N,) float16 range to the focal plane
    seed: Tensor      # (N,) int32  -- so validation splits are by SEED, never
                      #               by index: states from one flight are
                      #               correlated and would leak across a split
    round: Tensor     # (N,) int8   -- DAgger round; 1 = teacher-flown

    def __len__(self) -> int:
        return self.depth.shape[0]

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, f.name).numel() * getattr(self, f.name).element_size()
                   for f in fields(self))

    def to(self, device) -> "Rollouts":
        return Rollouts(**{f.name: getattr(self, f.name).to(device)
                           for f in fields(self)})

    def index(self, idx: Tensor) -> "Rollouts":
        return Rollouts(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    @staticmethod
    def cat(parts: list["Rollouts"]) -> "Rollouts":
        return Rollouts(**{f.name: torch.cat([getattr(p, f.name) for p in parts])
                           for f in fields(Rollouts)})

    def save(self, path: str) -> None:
        torch.save({f.name: getattr(self, f.name) for f in fields(self)}, path)

    @staticmethod
    def load(path: str) -> "Rollouts":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        # week-3 files predate the round field: they are all teacher-flown
        blob.setdefault("round", torch.ones(len(blob["seed"]), dtype=torch.int8))
        return Rollouts(**blob)

    def stats(self, cfg: Config) -> str:
        from sim.generate import STRATA
        lo, hi = cfg.decision_band
        band = (self.rng.float() >= lo) & (self.rng.float() <= hi)
        multi = self.n_modes >= 2
        out = [f"{len(self):,} states   multimodal {multi.float().mean():.4f}"
               f"   in band {band.float().mean():.4f}"]
        for k, name in enumerate(STRATA):
            m = self.stratum == k
            out.append(f"    {name:<10} n={int(m.sum()):>7,}  "
                       f"multi {multi[m].float().mean():.4f}   "
                       f"in-band multi {multi[m & band].float().mean():.4f}")
        return "\n".join(out)


def _pack(obs, lab, env, idx, cfg: Config, round_id: int) -> Rollouts:
    st = torch.cat([obs.vel, obs.goal_dir, obs.goal_dist], -1)
    return Rollouts(
        depth=obs.depth[idx].half().cpu(),
        state=st[idx].float().cpu(),
        traj=lab.traj[idx].float().cpu(),
        cost=lab.cost[idx].float().cpu(),
        valid=lab.valid[idx].cpu(),
        n_modes=lab.n_modes[idx].to(torch.int8).cpu(),
        stratum=env.scene.stratum[idx].to(torch.int8).cpu(),
        rng=(cfg.focal_x - env.pos[idx, 0]).half().cpu(),
        seed=env.scene.seeds[idx].to(torch.int32).cpu(),
        round=torch.full((len(idx),), round_id, dtype=torch.int8),
    )


def collect(cfg: Config, device, seeds, batch: int = 512, every: int = 1,
            progress: bool = False, policy_fn=None, round_id: int = 1) -> Rollouts:
    """Round 1: the teacher flies. Rounds 2+: the STUDENT flies and the teacher
    labels every state it wanders into (proposal §4.4).

    The labelling is the teacher's either way -- only the flying changes.
    """
    env = Env(cfg, device)
    pol = (teacher if policy_fn is None else policy_fn)(cfg)
    parts = []
    for i in range(0, len(seeds), batch):
        sc = generate(cfg, seeds[i:i + batch], device)
        obs = env.reset(sc)
        step = 0
        while not bool(env.done.all()):
            alive = ~env.done
            if step % every == 0 and bool(alive.any()):
                idx = alive.nonzero().flatten()
                parts.append(_pack(obs, label_state(env, cfg), env, idx, cfg,
                                   round_id))
            obs = env.step(pol(obs, env))
            step += 1
        if progress:
            n = sum(len(p) for p in parts)
            print(f"  seeds {i + len(sc.seeds):>6,}/{len(seeds):,}   states {n:>9,}")
    return Rollouts.cat(parts)


def sampler_weights(data: Rollouts, cfg: Config) -> Tensor | None:
    """Geometric reweighting only -- range to the focal plane, never n_modes.

    Returns None when band_oversample == 1.0, so the default path is plain
    uniform sampling with no weighting machinery in it at all.
    """
    if cfg.band_oversample == 1.0:
        return None
    lo, hi = cfg.decision_band
    r = data.rng.float()
    return torch.where((r >= lo) & (r <= hi),
                       torch.full_like(r, cfg.band_oversample),
                       torch.ones_like(r))
