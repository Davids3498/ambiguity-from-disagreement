"""Determinism is not a nicety: DAgger in week 4 reshuffles batches, and every
reported number is measured on held-out seeds that must mean the same
thing on every run."""
import subprocess
import sys

import torch

from sim.config import DEFAULT as CFG
from sim.env import Env
from sim.generate import generate, split_seeds
from eval.policies import reactive_depth, straight_line
from tests.util import DEV

HASH_SCRIPT = """
import hashlib, sys, torch
from sim.config import DEFAULT as CFG
from sim.env import Env
from sim.generate import generate, split_seeds
from eval.policies import reactive_depth
dev = "cuda" if torch.cuda.is_available() else "cpu"
env = Env(CFG, dev)
env.reset(generate(CFG, split_seeds("eval", 32), dev))
s = env.rollout(reactive_depth(CFG))
buf = b"".join(s[k].cpu().numpy().tobytes() for k in sorted(s))
print(hashlib.sha256(buf).hexdigest())
"""


def _run(policy_fn, n=32):
    env = Env(CFG, DEV)
    env.reset(generate(CFG, split_seeds("eval", n), DEV))
    return env.rollout(policy_fn(CFG))


def test_same_seeds_give_identical_rollouts():
    a, b = _run(reactive_depth), _run(reactive_depth)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_identical_across_process_restarts():
    """The strongest form: no cached state, no warm RNG, fresh interpreter."""
    out = [subprocess.run([sys.executable, "-c", HASH_SCRIPT], capture_output=True,
                          text=True, cwd=".", check=True).stdout.strip()
           for _ in range(2)]
    assert out[0] == out[1], out
    assert len(out[0]) == 64


def test_finished_episodes_are_frozen_not_resliced():
    """Masked stepping: once an env terminates its state must never move again."""
    env = Env(CFG, DEV)
    env.reset(generate(CFG, split_seeds("eval", 64), DEV))
    pol = straight_line(CFG)
    obs = env._observe()
    frozen = {}
    for _ in range(CFG.max_steps):
        done_before = env.done.clone()
        for i in done_before.nonzero().flatten().tolist():
            if i in frozen:
                assert torch.equal(frozen[i], env.pos[i]), f"env {i} moved after done"
            else:
                frozen[i] = env.pos[i].clone()
        if bool(done_before.all()):
            break
        obs = env.step(pol(obs))
    assert len(frozen) > 0


def test_batch_composition_does_not_change_an_episode():
    """Seed 2000005 must fly the same trajectory alone as in a crowd."""
    seeds = split_seeds("eval", 16)
    env = Env(CFG, DEV)
    env.reset(generate(CFG, seeds, DEV))
    big = env.rollout(reactive_depth(CFG))
    i = 5
    env.reset(generate(CFG, [seeds[i]], DEV))
    one = env.rollout(reactive_depth(CFG))
    for k in ("outcome", "steps"):
        assert int(one[k][0]) == int(big[k][i]), (k, one[k][0], big[k][i])
    assert abs(float(one["path_len"][0]) - float(big["path_len"][i])) < 1e-3
