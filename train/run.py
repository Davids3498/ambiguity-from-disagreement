"""python -m train.run --data data/round1.pt --loss rwta --steps 20000

Diagnostics are logged from optimiser step ONE, not step 100.
"""
import argparse
import csv
import os
import time

# must be set before the first CUDA context; cuBLAS reductions are otherwise
# nondeterministic and two same-seed runs diverge in the last few digits
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from sim.config import DEFAULT, Config, config_hash
from sim.generate import STRATA
from train import diagnostics as diag
from train.dataset import Rollouts, sampler_weights
from train.loss import losses, total
from train.student import Student

# A RANGE, not a threshold. Week 3 used `seed >= 2000`, which would have
# silently swallowed every DAgger round from week 4 onward into validation.
# This block preserves week 3's split exactly and leaves higher seeds free.
VAL_SEEDS = (2000, 2400)


def split(data: Rollouts):
    v = (data.seed >= VAL_SEEDS[0]) & (data.seed < VAL_SEEDS[1])
    return data.index(~v), data.index(v)


def log_fields(k: int) -> list[str]:
    return (["step", "loss", "loss_wp", "loss_cost", "val_loss",
             "diversity", "plaus_diversity"]
            + [f"win{j}" for j in range(k)]
            + [f"div_{s}" for s in STRATA])


def evaluate_probe(model, probe, cfg: Config, relax: float) -> dict:
    model.eval()
    with torch.no_grad():
        traj, cost = model(probe.depth, probe.state)
        wp, lc, winner = losses(traj, cost, probe.traj, probe.cost,
                                probe.valid, cfg, relax)
        row = {"val_loss": float(total(wp, lc, cfg)),
               "diversity": float(diag.diversity(traj)),
               "plaus_diversity": float(diag.plausible_diversity(
                   traj, cost, cfg.cost_margin))}
        wr = diag.win_rate(winner, probe.valid, model.k)
        row.update({f"win{j}": float(wr[j]) for j in range(model.k)})
        row.update({f"div_{s}": v for s, v in diag.by_stratum(
            traj, cost, probe.stratum.long(), cfg.cost_margin).items()})
    model.train()
    return row


def train(cfg: Config, data: Rollouts, device, loss_name: str, steps: int,
          k: int, seed: int, out: str, log_every: int, spread: float | None,
          dead_head: bool = False, probe_n: int = 4096, quiet: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Weeks 1 and 2 assert bitwise reproducibility; training holds the same
    # standard. cudnn's convolution backward is nondeterministic by default.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    if spread is not None:
        cfg = __import__("dataclasses").replace(cfg, head_init_spread=spread)
    relax = cfg.relax if loss_name == "rwta" else 0.0

    tr, va = split(data)
    # by round 4 the accumulated set is ~7 GiB; keep it on the host if it will
    # not comfortably fit alongside the model
    if tr.nbytes < 10 * 2**30:
        tr = tr.to(device)
    va = va.to(device)
    probe = va.index(torch.arange(min(probe_n, len(va)), device=device))
    w = sampler_weights(tr, cfg)
    w = w.to(device) if w is not None else None

    model = Student(cfg, k=k).to(device)
    if dead_head:
        # Park the last head 8 m to the side, far outside any teacher label
        # (|y| <= 3.5). Under WTA it can never be nearest, so it never receives
        # a gradient and never moves: a genuinely dead head.
        with torch.no_grad():
            model.out.bias.view(k, 10)[-1, [1, 4, 7]] += 8.0
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    os.makedirs(out, exist_ok=True)
    fh = open(os.path.join(out, "log.csv"), "w", newline="")
    wr = csv.DictWriter(fh, log_fields(k), lineterminator="\n")
    wr.writeheader()

    g = torch.Generator(device=device).manual_seed(seed)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        if w is None:
            idx = torch.randint(len(tr), (cfg.batch_size,), device=device, generator=g)
        else:
            idx = torch.multinomial(w, cfg.batch_size, replacement=True, generator=g)
        idx = idx.to(tr.depth.device)
        b = tr.index(idx).to(device)
        traj, cost = model(b.depth, b.state)
        l_wp, l_c, winner = losses(traj, cost, b.traj, b.cost, b.valid, cfg, relax)
        loss = total(l_wp, l_c, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # step ONE is logged, not step log_every
        if step == 1 or step % log_every == 0 or step == steps:
            row = {"step": step, "loss": float(loss),
                   "loss_wp": float(l_wp), "loss_cost": float(l_c)}
            row.update(evaluate_probe(model, probe, cfg, relax))
            wr.writerow({f: row.get(f, "") for f in log_fields(k)})
            fh.flush()
            if not quiet:
                wins = " ".join(f"{row[f'win{j}']:.2f}" for j in range(k))
                print(f"  step {step:>6}  loss {row['loss']:.4f}  "
                      f"val {row['val_loss']:.4f}  div {row['diversity']:.3f}m  "
                      f"plaus {row['plaus_diversity']:.3f}m  win [{wins}]")
    fh.close()
    dt = time.perf_counter() - t0
    ck = os.path.join(out, "model.pt")
    torch.save({"state_dict": model.state_dict(), "k": k, "device": str(device),
                "loss": loss_name, "seed": seed, "steps": steps,
                "config_hash": config_hash(cfg)}, ck)
    if not quiet:
        print(f"\n{steps:,} steps in {dt:.1f}s ({steps/dt:.0f} it/s) -> {ck}")
    return model, ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/round1.pt")
    ap.add_argument("--loss", choices=["wta", "rwta"], default="rwta")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--k", type=int, default=DEFAULT.k_heads)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/rwta")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--head-spread", type=float, default=None,
                    help="override symmetry breaking")
    ap.add_argument("--dead-head", action="store_true",
                    help="park the last head outside the label distribution")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Rollouts.load(args.data)
    tr, va = split(data)
    print(f"{args.loss.upper()}  K={args.k}  train {len(tr):,}  val {len(va):,}  "
          f"seed {args.seed}")
    train(DEFAULT, data, device, args.loss, args.steps, args.k, args.seed,
          args.out, args.log_every, args.head_spread, args.dead_head)


if __name__ == "__main__":
    main()
