#!/usr/bin/env python3
"""Learning curve: val MAE vs number of training FLOPS (the effective sample size, architecture.md §6.3 / §11).

If the error stops falling as flops are added, the net is limited by the labels; if it keeps falling, by quantity
(the trigger to consider the bootstrap, §6.3).  Same steps and hyper-parameters at every point.

  .venv/bin/python scripts/learning_curve.py --steps 3000 --fractions 0.125 0.25 0.5 1.0
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0])
    args = ap.parse_args()
    print(f"{'train flops':>12}{'train states':>14}{'val MAE %pot':>14}{'floor %pot':>12}")
    for f in args.fractions:
        run = subprocess.run([sys.executable, "-W", "ignore", os.path.join(HERE, "train.py"), "--steps", str(args.steps),
                              "--batch", str(args.batch), "--lr", str(args.lr), "--eval-every", str(args.steps),
                              "--flop-fraction", str(f), "--tag", f"_lc{f}"], capture_output=True, text=True)
        out = run.stdout
        m = re.search(r"FINAL train_states=(\d+) train_flops=(\d+) val_mae=([\d.]+) floor=([\d.]+)", out)
        print(f"{m.group(2):>12}{m.group(1):>14}{m.group(3):>14}{m.group(4):>12}" if m else f"fraction {f}: failed\n{(run.stderr or out)[-600:]}", flush=True)


if __name__ == "__main__":
    main()
