import ast
import pathlib

import torch

from sim.config import DEFAULT as CFG
from train.student import Student
from tests.util import DEV

FORBIDDEN = {"sim.scene", "sim.teacher", "sim.generate", "sim.env", "sim.cluster"}


def test_student_module_imports_nothing_privileged():
    """Proposal §4.3: 'Nothing privileged, ever.' Checkable by grep, not trust."""
    src = pathlib.Path("train/student.py").read_text()
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
    assert not (mods & FORBIDDEN), mods & FORBIDDEN


def test_forward_takes_only_depth_and_state():
    import inspect
    p = list(inspect.signature(Student.forward).parameters)
    assert p == ["self", "depth", "state"], p


def test_parameter_count_is_in_the_target_range():
    n = Student(CFG).n_params()
    assert 1_000_000 <= n <= 2_000_000, f"{n:,}"


def test_output_shapes():
    m = Student(CFG).to(DEV)
    t, c = m(torch.rand(5, CFG.cam_h, CFG.cam_w, device=DEV) * CFG.z_max,
             torch.randn(5, 7, device=DEV))
    assert t.shape == (5, CFG.k_heads, 3, 3) and c.shape == (5, CFG.k_heads)


def test_heads_are_distinct_at_initialisation():
    """Without this, WTA hands every label to head 0 and the rest never receive
    a gradient -- a dead run caused by initialisation, not by the loss."""
    from sim.cluster import pairwise
    m = Student(CFG).to(DEV)
    t, _ = m(torch.rand(64, CFG.cam_h, CFG.cam_w, device=DEV) * CFG.z_max,
             torch.randn(64, 7, device=DEV))
    d = pairwise(t)
    i, j = torch.triu_indices(CFG.k_heads, CFG.k_heads, 1, device=DEV)
    assert float(d[:, i, j].min()) > 0.05


def test_zero_spread_makes_heads_identical():
    from dataclasses import replace
    from sim.cluster import pairwise
    m = Student(replace(CFG, head_init_spread=0.0)).to(DEV)
    t, _ = m(torch.rand(8, CFG.cam_h, CFG.cam_w, device=DEV) * CFG.z_max,
             torch.randn(8, 7, device=DEV))
    i, j = torch.triu_indices(CFG.k_heads, CFG.k_heads, 1, device=DEV)
    assert float(pairwise(t)[:, i, j].max()) < 1e-6


def test_depth_normalisation_is_clamped_and_lives_in_the_model():
    """The simulator keeps metric depth (week 1); normalising is the student's
    input transform, so out-of-range input must not blow up the network."""
    m = Student(CFG).to(DEV)
    huge = torch.full((2, CFG.cam_h, CFG.cam_w), 1e4, device=DEV)
    t, c = m(huge, torch.zeros(2, 7, device=DEV))
    assert torch.isfinite(t).all() and torch.isfinite(c).all()


def test_eval_forward_is_deterministic():
    m = Student(CFG).to(DEV).eval()
    d = torch.rand(4, CFG.cam_h, CFG.cam_w, device=DEV) * CFG.z_max
    s = torch.randn(4, 7, device=DEV)
    with torch.no_grad():
        a, _ = m(d, s)
        b, _ = m(d, s)
    assert torch.equal(a, b)
