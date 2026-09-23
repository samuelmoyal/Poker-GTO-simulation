#!/usr/bin/env python3
"""Micro-benchmark of the label pipeline: state sampling from flop solves, then exact turn solves.

  model/.venv/bin/python bench/bench_labels.py [n_states=20]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from gtonet import labels, states  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    files = states.flop_files()
    t0 = time.time()
    sts = [s for p in files[: max(1, n // 4)] for s in states.flop_states(p, 4, seed=123)][:n]
    gen = time.time() - t0
    print(f"state sampling: {len(sts)} states from {len(files[: max(1, n // 4)])} flops in {gen:.2f}s "
          f"({gen / max(len(sts), 1) * 1000:.0f} ms/state)")
    times, expl = [], []
    with labels.LabelSolver() as solver:
        t0 = time.time()
        for s in sts:
            r = solver.solve(s.request(0.5))
            times.append(r["solve_s"])
            expl.append(r["expl_pct"])
        wall = time.time() - t0
    times.sort()
    print(f"exact turn solves at 0.5%: {len(sts) / wall:.2f} states/s wall | solve_s median {times[len(times) // 2]:.2f} "
          f"max {times[-1]:.2f} | expl% max {max(expl):.3f}")


if __name__ == "__main__":
    main()
