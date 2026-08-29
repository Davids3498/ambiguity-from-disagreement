"""Video: chase view | depth | telemetry.

The chase camera is the SAME ray tracer at a different pose and resolution --
there is no second geometry path, no renderer, no matplotlib. The drone is drawn
by projecting its position through the chase pinhole, which is cheaper than
adding a sphere kernel just to see the aircraft.

The panel layout has three slots on purpose. Week 5's per-frame disagreement
overlay drops into the telemetry slot; designing that in now costs nothing,
retrofitting it during the week-6 writeup costs an afternoon.
"""
import math

import imageio.v2 as imageio
import numpy as np
import torch

from sim.config import Config
from sim.depth import ray_dirs, rotate, trace
from sim.env import OUTCOME_NAME, Env
from sim.generate import SLOT, STRATA, generate

CHASE_W, CHASE_H = 384, 288
TELE_W = 200
CHASE_HFOV = 78.0
CHASE_FAR = 40.0

BG = (16, 18, 22)
INK = (232, 236, 242)
DIM = (120, 128, 140)
WARM = (232, 140, 72)
COOL = (72, 176, 176)

# 3x5 bitmap font: a table, not logic. Reused for week-5 overlay labels.
_FONT = {
    "0": "111101101101111", "1": "010110010010111", "2": "110001010100111",
    "3": "111001011001111", "4": "101101111001001", "5": "111100111001111",
    "6": "111100111101111", "7": "111001001001001", "8": "111101111101111",
    "9": "111101111001111", ".": "000000000000010", "-": "000000111000000",
    ":": "000010000010000", "/": "001001010100100", " ": "000000000000000",
    "A": "010101111101101", "B": "110101110101110", "C": "011100100100011",
    "D": "110101101101110", "E": "111100110100111", "F": "111100110100100",
    "G": "011100101101011", "H": "101101111101101", "I": "111010010010111",
    "J": "001001001101010", "K": "101101110101101", "L": "100100100100111",
    "M": "101111111101101", "N": "101111101101101", "O": "010101101101010",
    "P": "110101110100100", "Q": "010101101010001", "R": "110101110101101",
    "S": "011100010001110", "T": "111010010010010", "U": "101101101101010",
    "V": "101101101101010", "W": "101101111111101", "X": "101101010101101",
    "Y": "101101010010010", "Z": "111001010100111", "_": "000000000000111",
}


def draw_text(img, x, y, text, color, scale=2):
    for ch in str(text).upper():
        g = _FONT.get(ch)
        if g is not None:
            for r in range(5):
                for c in range(3):
                    if g[r * 3 + c] == "1":
                        img[y + r * scale:y + (r + 1) * scale,
                            x + c * scale:x + (c + 1) * scale] = color
        x += 4 * scale
    return x


_STOPS = np.array([[24, 26, 40], [38, 88, 148], [40, 158, 140],
                   [188, 190, 92], [242, 226, 200]], dtype=np.float32)


def colormap(t: np.ndarray) -> np.ndarray:
    """t in [0,1] -> RGB uint8, near = cool, far = pale."""
    t = np.clip(t, 0.0, 1.0) * (len(_STOPS) - 1)
    i = np.clip(t.astype(np.int32), 0, len(_STOPS) - 2)
    f = (t - i)[..., None]
    return (_STOPS[i] * (1 - f) + _STOPS[i + 1] * f).astype(np.uint8)


def _palette(scene) -> np.ndarray:
    nb, nc = scene.box_lo.shape[1], scene.cyl_c.shape[1]
    p = np.zeros((nb + nc, 3), np.float32)
    p[0] = (72, 66, 60)                       # floor
    p[1] = (44, 48, 55)                       # ceiling
    p[2] = p[3] = (56, 61, 70)                # side walls
    for k in range(SLOT["bar"], SLOT["distractor0"]):
        p[k] = WARM                           # every focal obstacle
    p[SLOT["distractor0"]:nb] = COOL          # distractors
    p[nb:] = WARM                             # the pole
    return p


def _project(pts_w, cam, yaw: float, pitch: float):
    """World points -> chase-camera (col, row, depth). The same pinhole the
    chase view is traced through, so overlays land where the geometry is."""
    v = pts_w - cam
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x1 = cy * v[:, 0] + sy * v[:, 1]
    y1 = -sy * v[:, 0] + cy * v[:, 1]
    z1 = v[:, 2]
    xc, zc = cp * x1 + sp * z1, -sp * x1 + cp * z1
    tan_h = math.tan(math.radians(CHASE_HFOV) / 2)
    tan_v = tan_h * CHASE_H / CHASE_W
    safe = np.maximum(xc, 1e-3)
    col = (-y1 / safe / tan_h + 1) * CHASE_W / 2 - 0.5
    row = (-zc / safe / tan_v + 1) * CHASE_H / 2 - 0.5
    return col, row, xc


def _polyline(img, col, row, ok, color, w=2):
    for i in range(len(col) - 1):
        if not (ok[i] and ok[i + 1]):
            continue
        n = max(2, int(max(abs(col[i + 1] - col[i]), abs(row[i + 1] - row[i]))))
        for t in np.linspace(0, 1, n):
            c = int(col[i] + t * (col[i + 1] - col[i]))
            r = int(row[i] + t * (row[i + 1] - row[i]))
            if 0 <= r < CHASE_H - w and 0 <= c < CHASE_W - w:
                img[r:r + w, c:c + w] = color


HEAD_COLS = [(120, 210, 255), (255, 170, 90), (150, 235, 150), (230, 140, 230)]


def draw_hypotheses(img, traj_body, cam, yaw, pitch, pos):
    """Draw the K predicted trajectories into the chase view.

    Two hypotheses separating over and under a bar is the whole thesis in one
    frame -- the number in the telemetry panel is only a summary of this.
    """
    for k in range(traj_body.shape[0]):
        wp = np.concatenate([np.zeros((1, 3)), traj_body[k]], 0)
        t = np.linspace(0, len(wp) - 1, 24)
        dense = np.stack([np.interp(t, np.arange(len(wp)), wp[:, d])
                          for d in range(3)], -1)
        cy, sy = math.cos(yaw), math.sin(yaw)
        world = np.stack([pos[0] + cy * dense[:, 0] - sy * dense[:, 1],
                          pos[1] + sy * dense[:, 0] + cy * dense[:, 1],
                          pos[2] + dense[:, 2]], -1)
        col, row, depth = _project(world, cam, yaw, pitch)
        _polyline(img, col, row, depth > 0.2, HEAD_COLS[k % len(HEAD_COLS)])


def chase_frame(scene1, pos, yaw, cfg: Config, dirs, pal, traj_body=None):
    """scene1 is a one-env Scene; pos (1,3); yaw (1,)."""
    dev = pos.device
    pitch = torch.full((1,), -0.26, device=dev)
    back = torch.stack([torch.cos(yaw), torch.sin(yaw),
                        torch.zeros_like(yaw)], -1)
    cam = pos - 6.5 * back + torch.tensor([0.0, 0.0, 2.4], device=dev)
    # keep the camera inside the corridor: outside it, the shell slabs occlude
    cam[:, 0] = cam[:, 0].clamp(0.25, cfg.length - 0.25)
    cam[:, 1] = cam[:, 1].clamp(-cfg.width / 2 + 0.25, cfg.width / 2 - 0.25)
    cam[:, 2] = cam[:, 2].clamp(0.25, cfg.height - 0.25)

    d_w = rotate(dirs, yaw, pitch)
    depth, ids = trace(scene1, cam.unsqueeze(1), d_w, CHASE_FAR,
                       cfg.prim_chunk, want_id=True)
    hit = cam.unsqueeze(1) + depth.unsqueeze(-1) * d_w      # (1,R,3)

    z = depth[0].cpu().numpy()
    i = ids[0].cpu().numpy()
    h = hit[0].cpu().numpy()

    img = np.empty((CHASE_H * CHASE_W, 3), np.float32)
    img[:] = np.array(BG, np.float32)
    seen = i >= 0
    base = pal[np.clip(i, 0, None)]
    # 3-D checker so forward motion is legible without any texture map
    chk = (np.floor(h[:, 0] / 2.0) + np.floor(h[:, 1] / 2.0)
           + np.floor(h[:, 2] / 2.0)) % 2.0
    shade = np.clip(1.35 - z / CHASE_FAR * 2.2, 0.18, 1.0) * (0.86 + 0.14 * chk)
    img[seen] = base[seen] * shade[seen, None]
    img = img.reshape(CHASE_H, CHASE_W, 3)

    # stamp the drone: project its world position through the same pinhole
    v = (pos - cam)[0]
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    x1 = cy * float(v[0]) + sy * float(v[1])
    y1 = -sy * float(v[0]) + cy * float(v[1])
    z1 = float(v[2])
    xc, zc = cp * x1 + sp * z1, -sp * x1 + cp * z1
    if xc > 0.2:
        tan_h = math.tan(math.radians(CHASE_HFOV) / 2)
        tan_v = tan_h * CHASE_H / CHASE_W
        col = int((-y1 / xc / tan_h + 1) * CHASE_W / 2 - 0.5)
        row = int((-zc / xc / tan_v + 1) * CHASE_H / 2 - 0.5)
        rad = max(3, int(90 / max(xc, 1.0)))
        yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
        disc = xx * xx + yy * yy <= rad * rad
        r0, r1 = max(0, row - rad), min(CHASE_H, row + rad + 1)
        c0, c1 = max(0, col - rad), min(CHASE_W, col + rad + 1)
        if r1 > r0 and c1 > c0:
            sub = disc[r0 - (row - rad):r1 - (row - rad),
                       c0 - (col - rad):c1 - (col - rad)]
            patch = img[r0:r1, c0:c1]
            patch[sub] = np.array(INK, np.float32) * 0.9
    img = img.astype(np.uint8)
    if traj_body is not None:
        draw_hypotheses(img, traj_body, cam[0].cpu().numpy(), float(yaw),
                        float(pitch), pos[0].cpu().numpy())
    return img


def depth_panel(depth: torch.Tensor, cfg: Config) -> np.ndarray:
    d = depth.cpu().numpy() / cfg.z_max
    rgb = colormap(d)
    sy, sx = CHASE_H // cfg.cam_h, CHASE_W // cfg.cam_w
    img = np.repeat(np.repeat(rgb, sy, 0), sx, 1)
    out = np.zeros((CHASE_H, CHASE_W, 3), np.uint8)
    out[:img.shape[0], :img.shape[1]] = img
    draw_text(out, 6, 6, "DEPTH 64X48", INK, 2)
    draw_text(out, 6, CHASE_H - 18, f"0-{cfg.z_max:.0f}M", DIM, 2)
    return out


def _bar(img, x, y, w, h, frac, color):
    img[y:y + h, x:x + w] = (34, 38, 44)
    n = int(np.clip(frac, 0.0, 1.0) * w)
    if n > 0:
        img[y:y + h, x:x + n] = color


def telemetry_panel(cfg, stratum, seed, t, speed, clear, prog, outcome=None,
                    disagree=None, k=0):
    img = np.zeros((CHASE_H, TELE_W, 3), np.uint8)
    img[:] = np.array((22, 25, 30), np.uint8)
    draw_text(img, 10, 12, STRATA[stratum], WARM, 3)
    draw_text(img, 10, 40, f"SEED {seed}", DIM, 2)
    y = 74
    for label, val, txt, frac, col in [
        ("TIME", t, f"{t:.2f}S", t / (cfg.max_steps * cfg.dt), (110, 130, 200)),
        ("SPEED", speed, f"{speed:.2f}", speed / cfg.v_max, (90, 190, 160)),
        ("CLEAR", clear, f"{min(clear, 9.99):.2f}", clear / 2.0,
         (210, 90, 80) if clear < cfg.r_drone * 2 else (90, 190, 160)),
        ("PROG", prog, f"{100 * prog:.0f}/100", prog, (200, 170, 90)),
    ]:
        draw_text(img, 10, y, label, DIM, 2)
        draw_text(img, 108, y, txt, INK, 2)
        _bar(img, 10, y + 14, TELE_W - 20, 7, frac, col)
        y += 38
    if disagree is not None:
        img[y + 2:y + 4, 10:TELE_W - 10] = (60, 66, 76)
        draw_text(img, 10, y + 12, "DISAGREEMENT", (232, 140, 72), 2)
        draw_text(img, 10, y + 30, f"{disagree:.3f} M", INK, 3)
        _bar(img, 10, y + 52, TELE_W - 20, 10, disagree / 0.40, (232, 140, 72))
        draw_text(img, 10, y + 68, f"{k} HYPOTHESES", DIM, 2)
    else:
        draw_text(img, 10, CHASE_H - 58, "SPEED M/S", DIM, 2)
        draw_text(img, 10, CHASE_H - 44, "CLEAR M", DIM, 2)
    if outcome is not None:
        col = (110, 200, 130) if outcome == "success" else (220, 96, 86)
        draw_text(img, 10, CHASE_H - 24, outcome, col, 3)
    return img


class Recorder:
    """env.rollout(policy, on_step=recorder) -- records env `idx` only."""

    def __init__(self, cfg: Config, scene, idx: int, writer, device, model=None):
        self.cfg, self.idx, self.writer, self.model = cfg, idx, writer, model
        self.scene1 = scene.index(idx)
        self.pal = _palette(scene)
        self.dirs = ray_dirs(CHASE_W, CHASE_H, CHASE_HFOV, device)
        self.last = None

    def _predict(self, obs):
        import torch
        from train.diagnostics import per_sample_disagreement
        i = self.idx
        with torch.no_grad():
            st = torch.cat([obs.vel, obs.goal_dir, obs.goal_dist], -1)[i:i + 1]
            traj, cost = self.model(obs.depth[i:i + 1], st)
            d = float(per_sample_disagreement(traj, cost, self.cfg.cost_margin)[0])
        return traj[0].cpu().numpy(), d

    def frame(self, env: Env, obs, outcome=None):
        i = self.idx
        pos, yaw = env.pos[i:i + 1], env.yaw[i:i + 1]
        tb, dis = self._predict(obs) if self.model is not None else (None, None)
        chase = chase_frame(self.scene1, pos, yaw, self.cfg, self.dirs, self.pal,
                            traj_body=tb)
        dep = depth_panel(obs.depth[i], self.cfg)
        tele = telemetry_panel(
            self.cfg, int(env.scene.stratum[i]), int(env.scene.seeds[i]),
            float(env.steps[i]) * self.cfg.dt, float(env.vel[i].norm()),
            float(env.scene.distance(env.pos[i:i + 1].unsqueeze(1))[0, 0]),
            float(env.pos[i, 0]) / self.cfg.length, outcome, disagree=dis,
            k=0 if self.model is None else self.model.k)
        self.last = np.concatenate([chase, dep, tele], axis=1)
        return self.last

    def __call__(self, env: Env, obs, action):
        if not bool(env.done[self.idx]):
            self.writer.append_data(self.frame(env, obs))


def record_episodes(cfg: Config, device, policy_fn, seeds, path: str,
                    hold: int = 14, model=None) -> None:
    """One mp4, episodes concatenated in the order given."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fps = int(round(1.0 / cfg.dt))
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as w:
        for seed in seeds:
            env = Env(cfg, device)
            scene = generate(cfg, [int(seed)], device)
            obs = env.reset(scene)
            rec = Recorder(cfg, scene, 0, w, device, model=model)
            env.rollout(policy_fn(cfg), on_step=rec)
            name = OUTCOME_NAME[int(env.outcome[0])]
            final = rec.frame(env, env._observe(), outcome=name)
            for _ in range(hold):
                w.append_data(final)
