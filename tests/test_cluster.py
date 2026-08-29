import math

import torch

from sim.cluster import components, pairwise, representatives
from tests.util import DEV


def _traj(pts):
    """(C,3) endpoints -> (1,C,3,3) trajectories with waypoints along the way."""
    e = torch.tensor(pts, device=DEV, dtype=torch.float32)
    return torch.stack([e / 3, 2 * e / 3, e], dim=1)[None]


def _all_active(t):
    return torch.ones(t.shape[:2], dtype=torch.bool, device=DEV)


def test_continuum_is_one_component():
    """A dense line of trajectories one grid step apart is ONE mode. This is the
    chaining behaviour single linkage is usually criticised for, and it is
    exactly what the wide_gap control stratum requires."""
    t = _traj([[5.0, -3.0 + 0.5 * i, 0.0] for i in range(13)])
    d = pairwise(t)
    _, n = components(d, _all_active(t), tau=0.65)
    assert int(n) == 1, int(n)


def test_two_separated_blobs_are_two_components():
    left = [[5.0, -3.0 + 0.3 * i, 0.0] for i in range(4)]
    right = [[5.0, 3.0 + 0.3 * i, 0.0] for i in range(4)]
    t = _traj(left + right)
    lab, n = components(pairwise(t), _all_active(t), tau=0.65)
    assert int(n) == 2, int(n)
    cost = torch.tensor([[1.0, 0.9, 1.1, 1.2, 0.5, 0.6, 0.7, 0.8]], device=DEV)
    _, c, v = representatives(t, cost, lab, _all_active(t), 4)
    assert int(v.sum()) == 2
    # one representative from each blob, each the cheapest of its own
    assert torch.allclose(c[0, :2].sort().values,
                          torch.tensor([0.5, 0.9], device=DEV))


def test_gap_just_below_tau_merges_and_just_above_splits():
    """Waypoints at e/3, 2e/3, e make RMS = sqrt(14/27) * endpoint separation,
    so tau=0.65 breaks at a 0.903 m endpoint gap."""
    crit = 0.65 / math.sqrt(14 / 27)
    for delta, expect in ((crit * 0.94, 1), (crit * 1.06, 2)):
        t = _traj([[5.0, 0.0, 0.0], [5.0, delta, 0.0]])
        _, n = components(pairwise(t), _all_active(t), tau=0.65)
        assert int(n) == expect, (delta, int(n))


def test_tau_zero_gives_one_component_per_survivor():
    t = _traj([[5.0, y, 0.0] for y in (-1.0, 0.0, 1.0)])
    _, n = components(pairwise(t), _all_active(t), tau=0.0)
    assert int(n) == 3


def test_inactive_entries_are_excluded():
    t = _traj([[5.0, -3.0, 0.0], [5.0, 0.0, 0.0], [5.0, 3.0, 0.0]])
    act = torch.tensor([[True, False, True]], device=DEV)
    _, n = components(pairwise(t), act, tau=0.65)
    assert int(n) == 2


def test_matches_independent_union_find():
    """Naive Python union-find on random point sets, deliberately unvectorised."""
    g = torch.Generator(device="cpu").manual_seed(0)
    for trial in range(12):
        c = 40
        e = torch.rand(c, 3, generator=g) * 6.0
        t = torch.stack([e / 3, 2 * e / 3, e], dim=1)[None].to(DEV)
        d = pairwise(t)
        tau = 0.5 + 0.1 * trial
        _, n = components(d, _all_active(t), tau=tau)

        parent = list(range(c))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        dm = d[0].cpu()
        for i in range(c):
            for j in range(i + 1, c):
                if dm[i, j] <= tau:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        assert int(n) == len({find(i) for i in range(c)}), (trial, tau)


def test_pairwise_is_rms_metres():
    a = _traj([[5.0, 0.0, 0.0]])
    b = _traj([[5.0, 3.0, 0.0]])
    t = torch.cat([a, b], dim=1)
    # waypoints differ by 1, 2 and 3 m -> RMS = sqrt((1+4+9)/3)
    assert abs(float(pairwise(t)[0, 0, 1]) - math.sqrt(14 / 3)) < 1e-5
