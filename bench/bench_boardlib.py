#!/usr/bin/env python3
"""Micro-benchmark of tools/boardlib (Rust) against the numpy / torch code it replaces.

  model/.venv/bin/python bench/bench_boardlib.py        (build first: cd tools/boardlib && cargo build --release)
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from gtonet import cards, equity, fastboard, poker  # noqa: E402


def timed(fn, reps=5):
    fn()
    t0 = time.time()
    for _ in range(reps):
        fn()
    return (time.time() - t0) / reps


def main():
    rng = np.random.default_rng(0)
    hands = np.array([rng.choice(52, 7, replace=False) for _ in range(63600)])       # one board's worth of 7-card hands
    t_np, t_rs = timed(lambda: poker.strength(hands)), timed(lambda: fastboard.strength(hands))
    print(f"7-card strength, 63.6k hands : numpy {t_np * 1e3:7.1f} ms | Rust {t_rs * 1e3:6.2f} ms  ({t_np / t_rs:5.0f}x, {63600 / t_rs / 1e6:.0f}M evals/s)")

    flop = ["9s", "7s", "2s"]
    turns = [c for c in cards.DECK if c not in flop]
    boards = [flop + [c] for c in turns]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    for b in boards[:4]:
        equity.BoardTables(b, dev, keep_w=False)
    t_tb = (time.time() - t0) / 4
    t0 = time.time()
    fb = fastboard.FastBoards(boards)
    t_fb = time.time() - t0
    t0 = time.time()
    wd = fb.win_tables()
    t_wd = time.time() - t0
    print(f"per-flop tables, 49 boards   : numpy+{dev.type} win table {t_tb * 49:6.1f} s (extrapolated from 4) | Rust boards {t_fb * 1e3:5.0f} ms + win tables {t_wd * 1e3:5.0f} ms")
    r = (rng.random((49, 10, cards.NUM_COMBOS)) * fb.hand_ok[:, None, :]).astype(np.float32)
    t_eq = timed(lambda: fb.equity(r))
    print(f"equity vs range by sorting   : {t_eq * 1e3:6.1f} ms for 490 ranges = {t_eq / 490 * 1e3:.3f} ms each (8 threads); no win table needed")
    wdt = torch.from_numpy(wd[:1]).to(dev)
    rt = torch.from_numpy(r[0]).to(dev)
    d = equity._const(dev)["disjoint"]

    def gpu():
        for _ in range(49):
            (rt @ wdt[0].T) / (rt @ d.T).clamp_min(1e-12)
        if dev.type == "mps":
            torch.mps.synchronize()
    print(f"equity vs range by table     : {timed(gpu) * 1e3:6.1f} ms for the same 490 ranges on {dev.type} (needs the tables above)")


if __name__ == "__main__":
    main()
