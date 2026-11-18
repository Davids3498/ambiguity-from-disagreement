"""The week-6 gate, made checkable.

"An honest positioning section was written" is not testable. "Every headline
number in the README matches results/sweep.csv" is.
"""
import csv
import importlib.util
import pathlib
import re

import numpy as np
import pytest

README = pathlib.Path("README.md")
SWEEP = pathlib.Path("results/sweep.csv")


def _text():
    return README.read_text()


def _sweep():
    if not SWEEP.exists():
        pytest.skip("sweep results not present")
    rows = list(csv.DictReader(open(SWEEP)))
    by = {}
    for r in rows:
        by.setdefault(r["cond"], []).append(r)
    return by


SCOPE_PARAGRAPHS = (
    "**Reproduced, not invented:** privileged teacher-student training, depth "
    "as the interface, receding-horizon trajectory output, multi-hypothesis "
    "prediction with per-hypothesis cost, DAgger.",
    "1. Head disagreement evaluated as an ambiguity *detector*",
    "2. Ablation of the loss variant itself — WTA vs. relaxed WTA",
    "3. K sweep against detection performance rather than against success rate alone.",
    "4. Diversity resolved by obstacle type, with a null baseline",
    "Claiming the multi-hypothesis architecture as a contribution would be "
    "caught by any reader who knows the paper. The contribution is the "
    "measurement layer.",
)


def test_scope_section_states_what_is_reproduced_and_what_is_new():
    """The scope statement must stay in the README, unedited."""
    t = _text()
    i = t.index("## Scope of the contribution")
    scope = t[i:t.index("## Results")]
    assert "**New here:**" in scope
    for p in SCOPE_PARAGRAPHS:
        assert p in scope, p[:70]


def test_every_reproduce_command_names_a_real_module():
    mods = set(re.findall(r"python -m ([\w.]+)", _text()))
    assert mods, "no commands found"
    for m in mods:
        assert importlib.util.find_spec(m) is not None, m


def test_ablation_numbers_match_the_sweep_csv():
    by = _sweep()
    t = _text()
    for cond, cols in (("single", ("success",)),
                       ("k2_rwta", ("success", "auc_bar", "auc_pole")),
                       ("k3_rwta", ("success", "auc_bar", "auc_pole")),
                       ("k6_wta", ("success",)),
                       ("null_k3", ("auc_bar", "auc_pole"))):
        for col in cols:
            v = np.mean([float(r[col]) for r in by[cond]])
            assert f"{v:.3f}" in t, f"{cond}.{col} = {v:.3f} missing from README"


def test_k_sweep_numbers_match():
    by = _sweep()
    t = _text()
    for k in (2, 3, 4, 6):
        rs = by[f"k{k}_rwta"]
        a = np.mean([(float(r["auc_bar"]) + float(r["auc_pole"])) / 2 for r in rs])
        assert f"{a:.3f}" in t, f"K={k} RWTA AUC {a:.3f} missing"


def test_null_baseline_appears_beside_every_bar_claim():
    """The bar margin is modest; the README must not quote it alone."""
    t = _text()
    by = _sweep()
    nb = np.mean([float(r["auc_bar"]) for r in by["null_k3"]])
    assert f"{nb:.3f}" in t
    i = t.index("Limitations")
    assert "never be quoted without the null" in t[i:]


def test_the_withdrawn_demo_claim_is_recorded_as_withdrawn():
    """Proposal §2 claims the agent slows at a bar. It does not."""
    t = _text()
    assert "withdrawn" in t.lower()
    assert "4.526" in t and "+0.002 m/s" in t


def test_dagger_null_is_reported():
    t = _text()
    assert "0.983 vs DAgger's 0.980" in t
    assert "sign flips" in t


def test_limitations_section_covers_every_required_item():
    t = _text()[_text().index("## Limitations"):]
    for phrase in ("definition, not an oracle", "clears the null",
                   "cannot be scored as detection rows", "is not a result",
                   "custom simulator", "modes come in pairs"):
        assert phrase in t, phrase


def test_headline_auc_is_the_pooled_number_not_the_all_states_one():
    """0.997 must never appear as a headline."""
    t = _text()
    head = t[:t.index("## Results")]
    assert "0.935" in head and "0.525" in head
    assert "0.997" not in head
