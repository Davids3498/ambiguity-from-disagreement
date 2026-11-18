"""The student. Sees depth, velocity and goal. Nothing privileged, ever.

That is enforced structurally: this module imports Config and torch, and
nothing from sim.scene, sim.teacher or sim.env. `tests/test_student.py` asserts
the import list, so the claim is checkable by grep rather than by trust.
"""
import torch
import torch.nn as nn
from torch import Tensor

from sim.config import Config


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(inplace=True),
    )


class Student(nn.Module):
    """K trajectory heads (3 waypoints x 3 dims) plus one scalar cost per head."""

    def __init__(self, cfg: Config, k: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.k_heads if k is None else k
        self.z_max = cfg.z_max
        self.conv = nn.Sequential(_block(1, 32), _block(32, 64),
                                  _block(64, 128), _block(128, 128))
        h, w = cfg.cam_h // 16, cfg.cam_w // 16
        self.trunk = nn.Sequential(
            nn.Linear(128 * h * w + 7, 512), nn.SiLU(inplace=True),
            nn.Linear(512, 256), nn.SiLU(inplace=True))
        self.out = nn.Linear(256, self.k * 10)
        self._init_heads()

    def _init_heads(self) -> None:
        """Straight-ahead prior plus a per-head random offset.

        Without the offset every head is identical, WTA hands every label to
        head 0 (argmin ties break by index), and heads 1..K-1 never receive a
        gradient -- a dead run from step one, caused by initialisation and not
        by the loss. The forced-collapse demonstration is built by
        setting head_init_spread to zero.
        """
        cfg = self.cfg
        nn.init.normal_(self.out.weight, std=0.01)
        if cfg.head_init_spread == 0.0:
            # The degenerate case, on purpose: zero weights as well as equal
            # biases makes every head EXACTLY identical, so WTA hands all labels
            # to head 0 and heads 1..K-1 never receive a gradient. This is used
            # to show the win-rate diagnostic registers the fault it exists
            # to catch. Never a default -- reachable only via --head-spread 0.
            nn.init.zeros_(self.out.weight)
        prior = torch.zeros(self.k, 10)
        for j in range(3):
            prior[:, 3 * j] = cfg.horizon * (j + 1) / 3.0        # forward
        g = torch.Generator().manual_seed(0)
        jitter = torch.randn(self.k, 3, 2, generator=g) * cfg.head_init_spread
        for j in range(3):
            prior[:, 3 * j + 1:3 * j + 3] += jitter[:, j] * (j + 1) / 3.0
        prior[:, 9] = 0.5                                        # cost prior
        with torch.no_grad():
            self.out.bias.copy_(prior.reshape(-1))

    def forward(self, depth: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """depth (N,H,W) metric metres, state (N,7) -> traj (N,K,3,3), cost (N,K).

        Normalisation lives here, in the input transform, not in the simulator:
        week 1 kept the depth buffer metric on purpose and that stays true.
        """
        x = (depth.unsqueeze(1).float() / self.z_max).clamp(0.0, 1.0)
        x = self.conv(x).flatten(1)
        y = self.out(self.trunk(torch.cat([x, state], -1))).view(-1, self.k, 10)
        return y[..., :9].view(-1, self.k, 3, 3), y[..., 9]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
