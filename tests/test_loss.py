import torch

from sim.config import DEFAULT as CFG
from train.loss import assign, losses, rms, weights
from tests.util import DEV


def _mk(vals):
    return torch.tensor(vals, device=DEV, dtype=torch.float32)


def test_assignment_matches_hand_computation():
    """Two labels, three heads, distances computed on paper."""
    pred = _mk([[[[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]],      # head 0: straight
                 [[1., 0., 1.], [2., 0., 2.], [3., 0., 3.]],      # head 1: up
                 [[1., 0., -1.], [2., 0., -2.], [3., 0., -3.]]]])  # head 2: down
    labels = _mk([[[[1., 0., .9], [2., 0., 1.9], [3., 0., 2.9]],   # near head 1
                   [[1., 0., -.8], [2., 0., -1.8], [3., 0., -2.8]]]])  # near head 2
    winner, d = assign(pred, labels)
    assert winner.tolist() == [[1, 2]]
    # head 1 vs label 0: displacements 0.1, 0.1, 0.1 -> RMS 0.1
    assert abs(float(d[0, 0, 1]) - 0.1) < 1e-5
    # head 0 vs label 0: 0.9, 1.9, 2.9 -> RMS sqrt((0.81+3.61+8.41)/3)
    assert abs(float(d[0, 0, 0]) - ((0.81 + 3.61 + 8.41) / 3) ** 0.5) < 1e-5


def test_one_head_may_win_several_labels():
    pred = _mk([[[[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]],
                 [[1., 9., 0.], [2., 9., 0.], [3., 9., 0.]]]])
    labels = _mk([[[[1., .1, 0.], [2., .1, 0.], [3., .1, 0.]],
                   [[1., -.1, 0.], [2., -.1, 0.], [3., -.1, 0.]]]])
    winner, _ = assign(pred, labels)
    assert winner.tolist() == [[0, 0]]


def test_wta_weights_are_one_hot_and_rwta_sums_to_one():
    w = torch.tensor([[0, 2]], device=DEV)
    assert torch.equal(weights(w, 3, 0.0),
                       torch.tensor([[[1., 0., 0.], [0., 0., 1.]]], device=DEV))
    r = weights(w, 3, 0.1)
    assert abs(float(r[0, 0, 0]) - 0.9) < 1e-6
    assert abs(float(r[0, 0, 1]) - 0.05) < 1e-6
    assert torch.allclose(r.sum(-1), torch.ones_like(r.sum(-1)))


def test_wta_gives_a_losing_head_exactly_zero_gradient():
    pred = _mk([[[[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]],
                 [[1., 5., 0.], [2., 5., 0.], [3., 5., 0.]]]]).requires_grad_(True)
    cost = torch.zeros(1, 2, device=DEV, requires_grad=True)
    labels = _mk([[[[1., .1, 0.], [2., .1, 0.], [3., .1, 0.]]]])
    lc = torch.zeros(1, 1, device=DEV)
    valid = torch.ones(1, 1, dtype=torch.bool, device=DEV)
    wp, c, _ = losses(pred, cost, labels, lc, valid, CFG, relax=0.0)
    (wp + c).backward()
    assert pred.grad[0, 1].abs().max() == 0.0
    assert pred.grad[0, 0].abs().max() > 0.0


def test_rwta_gradient_ratio_equals_the_weight_ratio():
    """Two heads equidistant from the label, so the only asymmetry is the RWTA
    weighting: the winner's gradient must be exactly (1-relax)/(relax/(K-1))
    times the loser's. Note a distant loser can carry a LARGER raw gradient than
    the winner -- small weight on a big error -- which is the mechanism that
    revives a dead head, so equal distance is the only clean place to check the
    weights themselves."""
    pred = _mk([[[[1., 1., 0.], [2., 1., 0.], [3., 1., 0.]],
                 [[1., -1., 0.], [2., -1., 0.], [3., -1., 0.]]]]).requires_grad_(True)
    cost = torch.zeros(1, 2, device=DEV, requires_grad=True)
    labels = _mk([[[[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]]])
    lc = torch.zeros(1, 1, device=DEV)
    valid = torch.ones(1, 1, dtype=torch.bool, device=DEV)
    wp, c, winner = losses(pred, cost, labels, lc, valid, CFG, relax=CFG.relax)
    (wp + c).backward()
    assert winner.tolist() == [[0]]
    win = float(pred.grad[0, 0].abs().max())
    lose = float(pred.grad[0, 1].abs().max())
    assert lose > 0.0
    assert abs(win / lose - (1 - CFG.relax) / CFG.relax) < 1e-4, (win, lose)


def test_wta_leaves_a_far_losing_head_untouched_but_rwta_pulls_it_back():
    """The reason RWTA exists: a head that has drifted far away still receives a
    gradient, so it can be revived instead of staying dead."""
    def grad(relax):
        pred = _mk([[[[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]],
                     [[1., 5., 0.], [2., 5., 0.], [3., 5., 0.]]]]).requires_grad_(True)
        cost = torch.zeros(1, 2, device=DEV, requires_grad=True)
        labels = _mk([[[[1., .1, 0.], [2., .1, 0.], [3., .1, 0.]]]])
        wp, c, _ = losses(pred, cost, labels, torch.zeros(1, 1, device=DEV),
                          torch.ones(1, 1, dtype=torch.bool, device=DEV), CFG, relax)
        (wp + c).backward()
        return float(pred.grad[0, 1].abs().max())
    assert grad(0.0) == 0.0
    assert grad(CFG.relax) > 0.0


def test_perfect_prediction_gives_zero_loss():
    labels = _mk([[[[1., 0., .5], [2., 0., 1.], [3., 0., 1.5]]]])
    pred = torch.cat([labels, labels + 5.0], dim=1)
    lc = torch.zeros(1, 1, device=DEV)
    cost = torch.zeros(1, 2, device=DEV)
    valid = torch.ones(1, 1, dtype=torch.bool, device=DEV)
    wp, c, _ = losses(pred, cost, labels, lc, valid, CFG, relax=0.0)
    assert float(wp) == 0.0 and float(c) == 0.0


def test_invalid_labels_do_not_contribute():
    labels = _mk([[[[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]],
                   [[9., 9., 9.], [9., 9., 9.], [9., 9., 9.]]]])
    pred = labels[:, :1].clone()
    lc = torch.zeros(1, 2, device=DEV)
    cost = torch.zeros(1, 1, device=DEV)
    valid = torch.tensor([[True, False]], device=DEV)
    wp, _, _ = losses(pred, cost, labels, lc, valid, CFG, relax=0.0)
    assert float(wp) == 0.0


def test_rms_is_metres_and_matches_the_cluster_metric():
    from sim.cluster import pairwise
    a = _mk([[[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]])
    b = _mk([[[1., 1., 0.], [2., 2., 0.], [3., 3., 0.]]])
    assert abs(float(rms(a, b)) - ((1 + 4 + 9) / 3) ** 0.5) < 1e-5
    t = torch.stack([a[0], b[0]])[None]
    assert abs(float(pairwise(t)[0, 0, 1]) - float(rms(a, b))) < 1e-5
