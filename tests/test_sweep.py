import csv
from dataclasses import replace

import numpy as np
import pytest
import torch

from eval.ablate import load
from eval.detect import auc
from eval.plots import roc_auc, roc_points
from sim.config import CONDITION_FIELDS, DEFAULT as CFG, config_hash
from sim.generate import split_seeds
from train.dataset import collect
from train.sweep import conditions, unimodal_only
from tests.util import DEV


# ---- the frozen config, made checkable ---------------------------------
def test_hash_ignores_exactly_the_condition_fields():
    base = config_hash(CFG)
    assert config_hash(replace(CFG, k_heads=6, relax=0.0)) == base
    for f in ("tau", "margin_abs", "horizon", "w_cost", "cost_margin", "lr"):
        assert config_hash(replace(CFG, **{f: getattr(CFG, f) * 1.1})) != base, f
    assert CONDITION_FIELDS == ("k_heads", "relax")


def test_analysis_refuses_to_mix_configs(tmp_path):
    p = tmp_path / "s.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["cond", "config_hash"], lineterminator="\n")
        w.writeheader()
        w.writerow({"cond": "a", "config_hash": config_hash(CFG)})
        w.writerow({"cond": "b", "config_hash": "deadbeefdeadbeef"})
    with pytest.raises(SystemExit, match="FROZEN CONFIG VIOLATED"):
        load(str(p))


def test_analysis_refuses_a_sweep_from_a_different_config(tmp_path):
    p = tmp_path / "s.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["cond", "config_hash"], lineterminator="\n")
        w.writeheader()
        w.writerow({"cond": "a", "config_hash": "0" * 16})
    with pytest.raises(SystemExit, match="current is"):
        load(str(p))


# ---- the null baseline --------------------------------------------------
def test_null_dataset_is_strictly_unimodal_and_keeps_most_states():
    """Same task minus the second mode -- not a different, smaller task."""
    d = collect(CFG, DEV, split_seeds("train", 60), batch=60)
    u = unimodal_only(d)
    assert int(u.n_modes.max()) == 1 and int(u.n_modes.min()) == 1
    assert len(u) / len(d) > 0.95, len(u) / len(d)


def test_null_condition_differs_from_the_headline_in_data_only():
    conds = {c[0]: c for c in conditions(seeds=(0,))}
    k3, null = conds["k3_rwta"], conds["null_k3"]
    assert (k3[1], k3[2]) == (null[1], null[2])      # same K, same loss
    assert k3[3] == "dagger" and null[3] == "null"   # only the data differs


def test_the_sweep_covers_the_planned_k_values():
    ks = {c[1] for c in conditions(seeds=(0,))}
    assert ks == {1, 2, 3, 4, 6}
    assert len(conditions(seeds=(0, 1, 2))) == 30


# ---- metrics ------------------------------------------------------------
def test_single_head_scores_exactly_one_half():
    """Zero disagreement means every pair ties, and a tie is worth 0.5. If this
    returned 0 or 1 the whole K sweep would be measuring a bug."""
    s = np.zeros(1000)
    y = np.zeros(1000, bool); y[:200] = True
    assert auc(s, y) == 0.5


def test_roc_area_equals_the_rank_based_auc_including_under_ties():
    rng = np.random.default_rng(0)
    s = np.r_[rng.normal(1, 1, 2000), rng.normal(0, 1, 5000)]
    y = np.r_[np.ones(2000, bool), np.zeros(5000, bool)]
    for score in (s, np.round(s, 1), np.round(s, 0)):
        f, t = roc_points(score, y)
        assert abs(roc_auc(f, t) - auc(score, y)) < 1e-9


def test_roc_starts_at_origin_ends_at_one_and_is_monotone():
    rng = np.random.default_rng(1)
    s = rng.normal(size=3000)
    y = rng.random(3000) < 0.3
    f, t = roc_points(s, y)
    assert (f[0], t[0]) == (0.0, 0.0)
    assert abs(f[-1] - 1.0) < 1e-12 and abs(t[-1] - 1.0) < 1e-12
    assert bool(np.all(np.diff(f) >= -1e-12) and np.all(np.diff(t) >= -1e-12))


def test_the_two_interval_types_measure_different_things():
    """Seed range and episode bootstrap must not be the same number."""
    from eval.ablate import _agg, _ci
    rs = [{"success": "0.90", "auc_bar_lo": "0.80", "auc_bar_hi": "0.95"},
          {"success": "0.94", "auc_bar_lo": "0.82", "auc_bar_hi": "0.96"}]
    m, half, _ = _agg(rs, "success")
    lo, hi = _ci(rs, "auc_bar")
    assert abs(m - 0.92) < 1e-9 and abs(half - 0.02) < 1e-9
    assert abs(lo - 0.81) < 1e-9 and abs(hi - 0.955) < 1e-9
