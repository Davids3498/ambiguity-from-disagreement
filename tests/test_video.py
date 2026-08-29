"""The video is the one artefact a reader cannot check, so the number drawn on
it must be the same number the ROC is computed from."""
import numpy as np
import torch

from sim.config import DEFAULT as CFG
from sim.env import Env
from sim.generate import generate, split_seeds
from train.diagnostics import per_sample_disagreement
from train.student import Student
from tests.util import DEV


def _recorder():
    from eval.record import Recorder
    m = Student(CFG, k=2).to(DEV).eval()
    sc = generate(CFG, split_seeds("eval", 1), DEV)
    env = Env(CFG, DEV)
    obs = env.reset(sc)
    return Recorder(CFG, sc, 0, None, DEV, model=m), env, obs, m


def test_overlay_disagreement_equals_the_scored_metric():
    rec, env, obs, m = _recorder()
    _, shown = rec._predict(obs)
    with torch.no_grad():
        st = torch.cat([obs.vel, obs.goal_dir, obs.goal_dist], -1)[:1]
        traj, cost = m(obs.depth[:1], st)
        ref = float(per_sample_disagreement(traj, cost, CFG.cost_margin)[0])
    assert shown == ref


def test_overlay_draws_one_polyline_per_head():
    rec, env, obs, m = _recorder()
    traj, _ = rec._predict(obs)
    assert traj.shape == (2, 3, 3)


def test_frames_are_the_expected_panel_layout():
    from eval.record import CHASE_H, CHASE_W, TELE_W
    rec, env, obs, m = _recorder()
    f = rec.frame(env, obs)
    assert f.shape == (CHASE_H, CHASE_W * 2 + TELE_W, 3)
    assert f.dtype == np.uint8


def test_projection_puts_the_origin_where_the_camera_looks():
    """A point straight ahead of the chase camera lands near the image centre."""
    from eval.record import CHASE_H, CHASE_W, _project
    cam = np.zeros(3)
    pts = np.array([[5.0, 0.0, 0.0]])
    col, row, d = _project(pts, cam, 0.0, 0.0)
    assert abs(col[0] - (CHASE_W / 2 - 0.5)) < 1e-6
    assert abs(row[0] - (CHASE_H / 2 - 0.5)) < 1e-6
    assert d[0] > 0
