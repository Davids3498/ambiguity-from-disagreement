"""Test scaffolding: build a Scene from plain lists, no generator involved."""
import torch

from sim.scene import Scene

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def scene_of(boxes, cyls=(), n=1, device=DEV, start=None, goal=None) -> Scene:
    """boxes: [(lo3, hi3), ...]   cyls: [(cx, cy, r), ...]"""
    if boxes:
        lo = torch.tensor([[b[0] for b in boxes]], device=device).expand(n, -1, 3)
        hi = torch.tensor([[b[1] for b in boxes]], device=device).expand(n, -1, 3)
        on = torch.ones(n, len(boxes), dtype=torch.bool, device=device)
    else:
        lo = hi = torch.zeros(n, 1, 3, device=device)
        on = torch.zeros(n, 1, dtype=torch.bool, device=device)
    if cyls:
        c = torch.tensor([[[x, y] for x, y, _ in cyls]], device=device).expand(n, -1, 2)
        r = torch.tensor([[cr for _, _, cr in cyls]], device=device).expand(n, -1)
        con = torch.ones(n, len(cyls), dtype=torch.bool, device=device)
    else:
        c = torch.zeros(n, 1, 2, device=device)
        r = torch.zeros(n, 1, device=device)
        con = torch.zeros(n, 1, dtype=torch.bool, device=device)
    def pt(v):
        return (torch.zeros(n, 3, device=device) if v is None
                else torch.tensor(v, device=device, dtype=torch.float32).expand(n, 3).clone())
    i = torch.zeros(n, dtype=torch.int64, device=device)
    return Scene(lo.contiguous(), hi.contiguous(), on, c.contiguous(),
                 r.contiguous(), con, pt(start), pt(goal), i, i.clone())
