#!/usr/bin/env python3
"""Train net_turn on the featurised turn-root dataset (MPS).

  .venv/bin/python scripts/train.py --overfit 16 --steps 300          # plumbing check: must reach ~0 error
  .venv/bin/python scripts/train.py --config small --steps 5000       # train split, validate on val

The first thing printed is the equity-baseline MAE on the validation set: the floor to beat (CLAUDE.md).
"""
import argparse
import math
import os
import sys
import time
import warnings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # any fallback warning is a bug: they are surfaced below

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import batching, dataset, evaluate, model as M  # noqa: E402


@torch.no_grad()
def state_errors(net, data, mode="canonical", bs=32) -> np.ndarray:
    """(N,) range-weighted MAE per state in % of pot (EV/pot units)."""
    net.eval()
    out = []
    for i in range(0, data.n, bs):
        b = data.batch(list(range(i, min(i + bs, data.n))), mode)
        o = net(b)
        e = 0.0
        for side in ("oop", "ip"):
            pred = M.to_evp(o[f"v_{side}"], b["spr"])
            e = e + (o[f"c_{side}"] * (pred - b[f"v_{side}"]).abs()).sum(-1) / 2
        out.append(100.0 * e.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small", choices=sorted(M.CONFIGS))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--mode", default="canonical", choices=["canonical", "augment", "identity"])
    ap.add_argument("--overfit", type=int, help="train and evaluate on the first N train states")
    ap.add_argument("--save", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flop-fraction", type=float, default=1.0, help="train on this share of the training flops (learning curve)")
    ap.add_argument("--tag", default="", help="checkpoint suffix")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    d_tr, d_va = dataset.load("train"), dataset.load("val" if not args.overfit else "train")
    if args.overfit:
        d_tr = {k: v[: args.overfit] for k, v in d_tr.items()}
        d_va = d_tr
    elif args.flop_fraction < 1.0:
        flops = sorted(set(d_tr["flop_key"]))
        keep = set(np.random.default_rng(0).permutation(flops)[: max(1, math.ceil(args.flop_fraction * len(flops)))])
        mask = np.array([f in keep for f in d_tr["flop_key"]])
        d_tr = {k: ([x for x, m in zip(v, mask) if m] if isinstance(v, list) else v[mask]) for k, v in d_tr.items()}
    tr, va = batching.TurnData(d_tr, dev), batching.TurnData(d_va, dev)
    print(f"device {dev} | train {tr.n} states / {len(set(d_tr['flop_key']))} flops | val {va.n} states / "
          f"{len(set(d_va['flop_key']))} flops", flush=True)
    floor = evaluate.state_mae_pct(*evaluate.equity_baseline(d_va), d_va)
    print(f"equity-baseline (floor) val MAE: {floor.mean():.3f}% of pot", flush=True)

    net = M.NetTurn(M.CONFIGS[args.config]).to(dev)
    print(f"{args.config}: {M.n_params(net) / 1e6:.2f}M params", flush=True)
    ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema), use_buffers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = args.warmup if args.warmup is not None else max(1, min(200, args.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    t0, run = time.time(), []
    for step in range(1, args.steps + 1):
        net.train()
        b = tr.batch(rng.choice(tr.n, min(args.batch, tr.n), replace=False), args.mode)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loss, parts = M.loss_fn(net(b), b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
        if step <= 3 and any("fall" in str(w.message).lower() for w in caught):
            print("!! MPS CPU-fallback warning:", [str(w.message)[:120] for w in caught], flush=True)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(), sched.step(), ema.update_parameters(net)
        run.append(float(loss))
        if step % args.eval_every == 0 or step == args.steps:
            err = state_errors(ema.module, va)
            print(f"step {step:5d} | loss {np.mean(run[-args.eval_every:]):.5f} | val MAE {err.mean():.3f}% "
                  f"(floor {floor.mean():.3f}%) | {(time.time() - t0) / step * 1000:.0f} ms/step", flush=True)
    os.makedirs(args.save, exist_ok=True)
    path = os.path.join(args.save, f"net_turn_{args.config}{args.tag}.pt")
    torch.save(dict(config=args.config, state=ema.module.state_dict()), path)
    print("saved", path)
    err = state_errors(ema.module, va)
    print(f"FINAL train_states={tr.n} train_flops={len(set(d_tr['flop_key']))} val_mae={err.mean():.3f} floor={floor.mean():.3f}")
    print("\nrange-weighted MAE, % of pot:\n" + evaluate.report(d_va, {"EQUITY floor": evaluate.equity_baseline(d_va),
                                                                      "net_turn (EMA)": _preds(ema.module, va)}))


@torch.no_grad()
def _preds(net, data, bs=32):
    """(pred_oop, pred_ip) in EV/pot, in the dataset's original hand order, for evaluate.report."""
    net.eval()
    po, pi = [], []
    for i in range(0, data.n, bs):
        idx = list(range(i, min(i + bs, data.n)))
        b = data.batch(idx, "canonical")
        o = net(b)
        for side, acc in (("oop", po), ("ip", pi)):
            acc.append(batching.unapply_perm(M.to_evp(o[f"v_{side}"], b["spr"]), b["perm"]).float().cpu().numpy())
    return np.concatenate(po), np.concatenate(pi)


if __name__ == "__main__":
    main()
