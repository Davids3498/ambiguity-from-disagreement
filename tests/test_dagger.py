import torch

from sim.config import DEFAULT as CFG
from sim.generate import split_seeds
from train.dagger import PER_ROUND, ROUND1_SEEDS, round_seeds
from train.dataset import Rollouts, collect
from train.run import VAL_SEEDS, split
from tests.util import DEV


def test_round_seeds_never_touch_validation():
    """Week 3 split validation with `seed >= 2000`, which would have swallowed
    every later round. The block must stay clean."""
    for r in range(2, 8):
        s = round_seeds(r)
        assert min(s) >= VAL_SEEDS[1], (r, min(s))
        assert not (set(s) & set(range(*VAL_SEEDS)))


def test_rounds_are_disjoint_from_each_other_and_from_round_one():
    seen = set(range(ROUND1_SEEDS))
    for r in range(2, 8):
        s = set(round_seeds(r))
        assert not (s & seen), r
        seen |= s


def test_week3_files_load_with_round_one():
    """The on-disk format gained a field; older files must still load."""
    import tempfile
    from dataclasses import fields
    d = collect(CFG, DEV, split_seeds("train", 8), batch=8)
    blob = {f.name: getattr(d, f.name) for f in fields(Rollouts) if f.name != "round"}
    with tempfile.NamedTemporaryFile(suffix=".pt") as fh:
        torch.save(blob, fh.name)
        e = Rollouts.load(fh.name)
    assert e.round.unique().tolist() == [1]
    assert len(e) == len(d)


def test_collect_uses_the_policy_it_is_given():
    """Rounds 2+ must be flown by the student, not the teacher."""
    seeds = split_seeds("train", 8)
    def parked(cfg):
        return lambda obs, env=None: torch.zeros(obs.vel.shape[0], 3,
                                                 device=obs.vel.device)
    a = collect(CFG, DEV, seeds, batch=8)
    b = collect(CFG, DEV, seeds, batch=8, policy_fn=parked, round_id=2)
    assert b.round.unique().tolist() == [2]
    assert len(a) != len(b) or not torch.equal(a.state, b.state)


def test_dataset_accumulates_and_never_replaces():
    seeds = split_seeds("train", 8)
    a = collect(CFG, DEV, seeds, batch=8)
    b = collect(CFG, DEV, split_seeds("train", 8, ROUND1_SEEDS), batch=8, round_id=2)
    c = Rollouts.cat([a, b])
    assert len(c) == len(a) + len(b)
    assert torch.equal(c.depth[:len(a)], a.depth)
    assert set(c.round.unique().tolist()) == {1, 2}


def test_split_puts_later_rounds_in_train_not_validation():
    a = collect(CFG, DEV, split_seeds("train", 8, 2000), batch=8)          # val block
    b = collect(CFG, DEV, split_seeds("train", 8, ROUND1_SEEDS), batch=8, round_id=2)
    tr, va = split(Rollouts.cat([a, b]))
    assert set(va.round.unique().tolist()) == {1}
    assert set(tr.round.unique().tolist()) == {2}


def test_shift_axes_do_not_read_the_label():
    """The distribution-shift diagnostic must not depend on n_modes, or it
    cannot be used to argue about the label distribution."""
    import ast
    import pathlib
    from eval.shift import axes
    d = collect(CFG, DEV, split_seeds("train", 8), batch=8)
    a = axes(d, CFG)
    d2 = Rollouts(**{f: (torch.zeros_like(getattr(d, f)) if f == "n_modes"
                         else getattr(d, f))
                     for f in d.__dataclass_fields__})
    a2 = axes(d2, CFG)
    for k in ("speed", "goal_offaxis", "teacher_cost"):
        assert torch.equal(a[k], a2[k]), k
