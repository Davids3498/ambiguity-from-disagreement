import torch

from sim.config import DEFAULT as CFG
from sim.generate import split_seeds
from train.dataset import Rollouts, collect, sampler_weights
from tests.util import DEV


def _small():
    return collect(CFG, DEV, split_seeds("train", 24), batch=24)


def test_roundtrip_preserves_every_field_bitwise(tmp_path):
    d = _small()
    p = str(tmp_path / "d.pt")
    d.save(p)
    e = Rollouts.load(p)
    from dataclasses import fields
    for f in fields(Rollouts):
        assert torch.equal(getattr(d, f.name), getattr(e, f.name)), f.name


def test_multimodal_fraction_is_consistent_with_the_week2_audit():
    d = _small()
    lo, hi = CFG.decision_band
    band = (d.rng.float() >= lo) & (d.rng.float() <= hi)
    multi = d.n_modes >= 2
    amb = (d.stratum == 0) | (d.stratum == 1)
    ctl = (d.stratum == 2) | (d.stratum == 3)
    assert 0.25 < float(multi[amb & band].float().mean()) < 0.75
    assert float(multi[ctl & band].float().mean()) < 0.05


def test_whole_flight_is_overwhelmingly_unimodal():
    """The number that shapes week 3's whole design: a naive round-1 dataset is
    ~99% unimodal, so the disagreement signal rides on a thin slice."""
    d = _small()
    assert float((d.n_modes >= 2).float().mean()) < 0.05


def test_validation_split_is_by_seed_not_by_index():
    """States within one flight are correlated; an index split would leak."""
    from train.run import VAL_SEEDS, split
    d = collect(CFG, DEV, split_seeds("train", 40, 1990), batch=40)
    tr, va = split(d)
    assert len(tr) > 0 and len(va) > 0
    assert not (set(tr.seed.tolist()) & set(va.seed.tolist()))


def test_default_sampling_is_uniform_with_no_weighting_machinery():
    """band_oversample defaults to 1.0 and must return None, so the default path
    has no reweighting in it at all."""
    d = _small()
    assert CFG.band_oversample == 1.0
    assert sampler_weights(d, CFG) is None


def test_reweighting_when_enabled_uses_geometry_not_n_modes():
    from dataclasses import replace
    d = _small()
    w = sampler_weights(d, replace(CFG, band_oversample=4.0))
    lo, hi = CFG.decision_band
    band = (d.rng.float() >= lo) & (d.rng.float() <= hi)
    assert torch.equal(w[band], torch.full_like(w[band], 4.0))
    assert torch.equal(w[~band], torch.ones_like(w[~band]))
    # crucially: unimodal in-band states are up-weighted exactly as much as
    # multimodal ones -- the weight cannot see n_modes
    uni = band & (d.n_modes < 2)
    mul = band & (d.n_modes >= 2)
    if bool(uni.any()) and bool(mul.any()):
        assert float(w[uni].mean()) == float(w[mul].mean())


def test_index_and_cat_are_consistent():
    d = _small()
    a = d.index(torch.arange(10))
    b = d.index(torch.arange(10, 20))
    c = Rollouts.cat([a, b])
    assert torch.equal(c.depth, d.index(torch.arange(20)).depth)
