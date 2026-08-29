import torch

from sim.config import DEFAULT as CFG
from sim.generate import split_seeds
from train.dataset import Rollouts, collect
from train.run import train
from tests.util import DEV


def _data():
    return collect(CFG, DEV, split_seeds("train", 60, 1980), batch=60)


def _run(seed, out, **kw):
    return train(CFG, _data(), DEV, "rwta", 30, CFG.k_heads, seed, out, 10,
                 None, quiet=True, probe_n=256, **kw)


def test_same_seed_gives_the_same_model(tmp_path):
    a, _ = _run(0, str(tmp_path / "a"))
    b, _ = _run(0, str(tmp_path / "b"))
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_different_seeds_give_different_models(tmp_path):
    a, _ = _run(0, str(tmp_path / "a"))
    b, _ = _run(1, str(tmp_path / "b"))
    assert not all(torch.equal(pa, pb)
                   for pa, pb in zip(a.parameters(), b.parameters()))


def test_diagnostics_exist_at_step_one(tmp_path):
    """Proposal §6: logged from step one, not from step 100."""
    import csv
    out = str(tmp_path / "r")
    _run(0, out)
    rows = list(csv.DictReader(open(f"{out}/log.csv")))
    assert rows[0]["step"] == "1"
    for f in ("diversity", "plaus_diversity", "win0", "win1", "win2"):
        assert rows[0][f] != "", f
        assert float(rows[0][f]) == float(rows[0][f])       # not NaN


def test_a_parked_head_stays_dead_under_wta_and_revives_under_rwta(tmp_path):
    """The instrument must register the fault it exists to catch (plan3 §8).

    A head parked outside the label distribution can never be nearest, so under
    WTA it never receives a gradient. Note that diversity RISES in this case --
    which is exactly why win rate is logged alongside it."""
    import csv
    data = _data()
    for loss, expect_dead in (("wta", True), ("rwta", False)):
        out = str(tmp_path / loss)
        train(CFG, data, DEV, loss, 400, 3, 0, out, 200, None,
              dead_head=True, quiet=True, probe_n=256)
        last = list(csv.DictReader(open(f"{out}/log.csv")))[-1]
        dead = float(last["win2"]) == 0.0
        assert dead == expect_dead, (loss, last["win2"])
        if expect_dead:
            assert float(last["diversity"]) > 3.0, last["diversity"]
