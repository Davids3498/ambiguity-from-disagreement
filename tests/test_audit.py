"""The gate itself, as a test, so it cannot silently rot in week 3+."""
import numpy as np
import torch

from eval.audit import approach_states, frac_multi, sweep_states
from sim.config import DEFAULT as CFG
from sim.generate import generate, split_seeds
from sim.teacher import plan
from tests.util import DEV


def _rows(n=200, **kw):
    sc = generate(CFG, split_seeds("eval", n), DEV)
    return sweep_states(sc, CFG, DEV, **kw)


def test_gate_ambiguous_strata_are_multimodal_in_band():
    rows = _rows()
    for s in ("bar", "pole"):
        assert frac_multi(rows, s, CFG.decision_band) >= 0.60, s


def test_gate_control_strata_are_not():
    """The half proposal §7 does not state. A teacher returning two modes
    everywhere would pass the multimodality test and be worthless."""
    rows = _rows()
    for s in ("window", "wide_gap"):
        assert frac_multi(rows, s, CFG.decision_band) <= 0.05, s


def test_separation_survives_the_tau_sweep():
    for tau in (0.5, 0.65, 0.8):
        rows = _rows(tau=tau)
        amb = min(frac_multi(rows, s, CFG.decision_band) for s in ("bar", "pole"))
        ctl = max(frac_multi(rows, s, CFG.decision_band)
                  for s in ("window", "wide_gap"))
        assert amb >= 0.55 and ctl <= 0.05, (tau, amb, ctl)


def test_ambiguity_vanishes_beyond_the_horizon():
    """Beyond horizon+0.5 the candidates stop short of the obstacle, so there is
    genuinely no decision in the plan yet. If this is nonzero, something other
    than the obstacle is splitting the modes."""
    rows = _rows()
    far = [r for r in rows if r["range"] >= CFG.horizon + 1.0]
    assert np.mean([r["n_modes"] >= 2 for r in far]) < 0.02


def test_audit_rows_are_deterministic():
    a, b = _rows(n=64), _rows(n=64)
    assert a == b


def test_mode_cap_is_not_binding():
    rows = _rows()
    assert np.mean([r["n_modes"] > CFG.max_modes for r in rows]) < 0.01
