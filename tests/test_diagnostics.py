import torch

from sim.cluster import pairwise
from train import diagnostics as diag
from tests.util import DEV


def _heads(offsets):
    """offsets: list of lateral y offsets -> (1,K,3,3)."""
    return torch.stack([torch.tensor([[1., y, 0.], [2., y, 0.], [3., y, 0.]],
                                     device=DEV) for y in offsets])[None]


def test_win_rate_sums_to_one():
    w = torch.tensor([[0, 2, 1]], device=DEV)
    v = torch.ones(1, 3, dtype=torch.bool, device=DEV)
    r = diag.win_rate(w, v, 3)
    assert abs(float(r.sum()) - 1.0) < 1e-6


def test_win_rate_ignores_invalid_labels():
    w = torch.tensor([[0, 1]], device=DEV)
    v = torch.tensor([[True, False]], device=DEV)
    assert diag.win_rate(w, v, 3).tolist() == [1.0, 0.0, 0.0]


def test_win_rate_detects_a_dead_head():
    """The fault the diagnostic exists to catch."""
    w = torch.zeros(64, 2, dtype=torch.long, device=DEV)
    v = torch.ones(64, 2, dtype=torch.bool, device=DEV)
    r = diag.win_rate(w, v, 3)
    assert r.tolist() == [1.0, 0.0, 0.0]


def test_diversity_is_zero_for_identical_heads():
    assert float(diag.diversity(_heads([0.0, 0.0, 0.0]))) < 1e-6


def test_diversity_matches_the_cluster_metric_in_metres():
    h = _heads([0.0, 1.0])
    assert abs(float(diag.diversity(h)) - float(pairwise(h)[0, 0, 1])) < 1e-6
    assert abs(float(diag.diversity(h)) - 1.0) < 1e-5      # metres


def test_diversity_is_high_for_a_dead_head_which_is_why_win_rate_is_needed():
    """A head parked 8 m away makes diversity look excellent while it is in fact
    garbage. Proposal §6: neither diagnostic is sufficient alone."""
    h = _heads([0.0, 0.1, 8.0])
    assert float(diag.diversity(h)) > 4.0


def test_cost_plausible_diversity_excludes_an_implausible_head():
    h = _heads([0.0, 0.1, 8.0])
    cheap = torch.tensor([[0.0, 0.02, 0.0]], device=DEV)     # all plausible
    dear = torch.tensor([[0.0, 0.02, 5.0]], device=DEV)      # third one is not
    assert float(diag.plausible_diversity(h, cheap, 0.15)) > 4.0
    assert float(diag.plausible_diversity(h, dear, 0.15)) < 0.2


def test_single_head_has_zero_diversity_by_definition():
    h = _heads([0.0])
    c = torch.zeros(1, 1, device=DEV)
    assert float(diag.diversity(h)) == 0.0
    assert float(diag.plausible_diversity(h, c, 0.15)) == 0.0
    assert float(diag.per_sample_disagreement(h, c, 0.15)[0]) == 0.0


def test_per_sample_disagreement_agrees_with_the_batch_mean():
    h = _heads([0.0, 0.5, 1.5]).repeat(7, 1, 1, 1)
    c = torch.zeros(7, 3, device=DEV)
    s = diag.per_sample_disagreement(h, c, 0.15)
    assert abs(float(s.mean()) - float(diag.plausible_diversity(h, c, 0.15))) < 1e-6
