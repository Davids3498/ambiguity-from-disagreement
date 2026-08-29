# Pre-flight check, between weeks 3 and 4

Two questions were worth answering before spending a week on DAgger. Both are
cheap. Neither is a week-5 result: single seed per condition, config not frozen,
no K sweep, and the distribution is wrong (see *Caveats*).

---

## 1. Is the task saturated? (proposal §7)

The risk table says: tune obstacle density until the single-head baseline lands
well below ceiling, then freeze. Measured on 400 held-out EVAL seeds:

| | bar | pole | window | wide_gap | ALL |
|---|---|---|---|---|---|
| **K=1** (single regression head) | 0.820 | **0.640** | 0.960 | 1.000 | 0.855 |
| K=3 WTA | 0.950 | 0.860 | 0.850 | 0.980 | 0.910 |
| K=3 RWTA | 0.970 | 0.920 | 0.960 | 1.000 | 0.963 |
| teacher | 1.000 | 1.000 | 0.980 | 1.000 | 0.995 |

**No density tuning needed.** The single-head baseline sits at 0.855 overall and
0.640 on `pole` — far from ceiling.

Better: the multi-head gain is concentrated exactly where the theory says it must
be. K=1 → K=3 RWTA moves `pole` by **+0.28** and `bar` by **+0.15**, while the
unimodal controls do not move at all (`window` 0.960 → 0.960, `wide_gap` 1.000 →
1.000). That is the mode-averaging signature of proposal §1 — a single MSE head
averaging over/under and left/right into the obstacle — appearing only on the
strata that have two modes to average.

## 2. Does disagreement actually detect ambiguity? (proposal §2)

ROC AUC of cost-plausible head disagreement against the teacher's
`n_modes ≥ 2`, on 54,037 validation states from 400 held-out seeds. Bootstrap
CIs resample **seeds**, not states.

| | K=1 | K=3 WTA | K=3 RWTA |
|---|---|---|---|
| decision band, pooled | 0.500 | 0.882 [0.862, 0.901] | **0.983** [0.975, 0.990] |
| band, `bar` (n₊=305) | 0.500 | 0.651 [0.591, 0.704] | **0.934** [0.906, 0.959] |
| band, `pole` (n₊=346) | 0.500 | 0.693 [0.638, 0.737] | **0.973** [0.959, 0.986] |

**The central claim has support.** Disagreement alone separates genuinely
multimodal states from unimodal ones at 0.93–0.97 per stratum, on held-out
seeds, with no map at inference.

K=1 scores exactly 0.500 — it has identically zero disagreement, so every pair
ties. That is the right answer and it also confirms the AUC implementation
handles ties correctly rather than silently scoring 0 or 1.

**The WTA vs RWTA gap is large** — 0.934 vs 0.651 on `bar`, non-overlapping CIs.
That is proposal §3's second claimed contribution (an ablation the source paper
does not perform), and on this evidence it is a real effect rather than a
formality.

---

## Caveats, all of which week 5 must fix

1. **The "all states" AUC of 0.997 is inflated and must not be quoted.** 98.8% of
   states are far from any obstacle, where the answer is trivially "unimodal";
   that number largely measures *is the drone near something*. The honest
   headline is the in-band, per-stratum row.
2. **`window` and `wide_gap` cannot carry their own AUC.** In-band they have 2
   and 0 positive examples. They contribute to detection as the **negative
   class**, not as separate detection rows, and the report now flags any row
   resting on fewer than 20 positives.
3. **Wrong distribution.** This is scored on round-1 states, which the *teacher*
   flew. The deployment condition is the student flying. Week 5's headline must
   be measured on student-visited states from held-out eval seeds.
4. **One seed per condition.** No seed-to-seed variation measured. The CIs above
   are over episodes, not over training runs.

## What this changes about weeks 4–6

- **Week 5's config can be frozen as it stands.** §7's density gate is satisfied
  without intervention.
- **Week 4's gate should be reframed.** "Success rate improves" has little room
  left (0.963 → 0.995 ceiling). The question DAgger actually poses now is
  whether the detector *survives the distribution shift* to student-visited
  states — which is caveat 3, and is the thing week 5 depends on.
- **Week 5's machinery largely exists**: `eval/detect.py` has AUC, tie handling
  and a seed-level bootstrap; `train/diagnostics.py` has the per-state score.
  What is missing is the K sweep, the ROC curves, and repeated seeds.
