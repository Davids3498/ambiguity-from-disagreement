"""All magic numbers live here. Frozen: a Config is a value, not a mutable global.

Week 1 defaults are a starting point, NOT frozen science. Obstacle density is
tuned in week 5 against the single-head baseline and frozen there,
before any condition is run.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- corridor (world frame: x forward, y left, z up) ---
    length: float = 30.0
    width: float = 8.0
    height: float = 5.0

    # --- dynamics ---
    dt: float = 0.05          # 20 Hz; sim step == control step
    v_max: float = 5.0
    a_max: float = 8.0
    tau: float = 0.15         # first-order actuator lag
    r_drone: float = 0.30     # collision radius
    max_steps: int = 400      # 20 s timeout

    # --- camera ---
    cam_w: int = 64
    cam_h: int = 48
    hfov_deg: float = 90.0
    z_max: float = 12.0       # depth clip

    # --- scene layout ---
    focal_x: float = 15.0     # focal obstacle sits mid-corridor
    focal_band: float = 4.0   # no distractors within +/- this of focal_x
    bar_thick: float = 0.60   # 3.7 px at 6 m (1.54 deg/px vertical)
    bar_gap_min: float = 1.20 # guaranteed clearance above AND below
    focal_offset: float = 0.50  # focal obstacle offset from the approach line
    pole_r: float = 0.35
    pole_y_max: float = 1.50
    window_ap: float = 1.40   # square aperture side
    win_off_lat: float = 1.20 # aperture offset from the approach line...
    win_off_vert: float = 0.70  # ...bounded by what one horizon can translate
    wall_thick: float = 0.30
    gap_width: float = 3.75   # wide-gap stratum central opening
    n_distractor_max: int = 6
    distractor_size: float = 0.60

    # --- batching ---
    prim_chunk: int = 8       # primitives per render chunk; see sim/depth.py
    # every env shares one slot layout, so the active mask IS the padding:
    # no max-primitive count is needed anywhere.

    # --- teacher (week 2) ---
    horizon: float = 5.0      # fixed forward horizon; varying it would make
                              # candidate costs incomparable
    v_des: float = 4.5        # nominal planning speed
    n_lat: int = 15
    n_vert: int = 13
    lat_max: float = 3.5
    vert_max: float = 2.5
    n_samples: int = 12       # spline samples for the clearance integral
    d_safe: float = 1.0       # clearance below which the penalty starts
    w_clear: float = 1.0
    w_goal: float = 1.0
    w_smooth: float = 0.3
    margin_abs: float = 0.12  # hybrid survivor threshold; see sim/teacher.py
    margin_rel: float = 0.15
    tau: float = 0.65         # cluster threshold, metres RMS; flat
                              # plateau 0.5-0.8, merges modes above 1.0
    max_modes: int = 4
    cand_chunk: int = 16      # candidates per clearance chunk

    # --- student / training (week 3) ---
    k_heads: int = 3          # source paper's K; week 5 sweeps {1,2,3,4,6}
    w_cost: float = 0.05      # cost-head weight; waypoint term is /horizon^2
    relax: float = 0.10       # RWTA: winner 1-relax, losers relax/(K-1)
    head_init_spread: float = 0.5   # metres; symmetry breaking
    cost_margin: float = 0.15 # "cost-plausible" band for head disagreement
    band_oversample: float = 1.0    # 1.0 = uniform. Geometry only, never n_modes
    lr: float = 3e-4
    batch_size: int = 256

    # --- control (shared by teacher and, in week 3, the student) ---
    kv: float = 8.0           # velocity P gain; 4.0 could not track a plan
                              # through a 1.4 m aperture (found in week 2)

    # --- geometry sanity (asserted in tests/test_generate.py) ---
    decision_range: float = 6.0    # where the planner must already see the mode
    min_subtend_px: float = 3.0    # focal obstacle height at decision_range

    # --- goal / start ---
    start_x: float = 1.0
    goal_margin: float = 1.0  # success when x > length - goal_margin

    @property
    def n_rays(self) -> int:
        return self.cam_w * self.cam_h

    @property
    def decision_band(self) -> tuple[float, float]:
        """Ranges at which the candidate endpoints straddle the focal obstacle.

        Derived from the horizon, not chosen: beyond horizon+0.5 the candidates
        stop short of the obstacle and there is genuinely no decision in the plan
        yet; inside horizon-1 the drone is already committed.
        """
        return (self.horizon - 1.0, self.horizon + 0.5)


CONDITION_FIELDS = ("k_heads", "relax")


def config_hash(cfg: Config) -> str:
    """Identity of everything week 5 holds frozen.

    k_heads and relax are excluded because those ARE the conditions. Any other
    difference between two runs makes them incomparable, and eval/ablate.py
    refuses to build a table across mismatched hashes rather than trusting that
    nobody edited the config mid-sweep.
    """
    import dataclasses
    import hashlib
    d = sorted((f.name, getattr(cfg, f.name)) for f in dataclasses.fields(cfg)
               if f.name not in CONDITION_FIELDS)
    return hashlib.sha256(repr(d).encode()).hexdigest()[:16]


DEFAULT = Config()
