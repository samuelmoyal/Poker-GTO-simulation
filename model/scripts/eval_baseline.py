#!/usr/bin/env python3
"""MAE of the parameter-free baselines: the floor any learned model must beat (CLAUDE.md, before training)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import dataset, evaluate  # noqa: E402


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else None
    d = dataset.load(split)
    print(f"{len(d['id'])} states ({split or 'all splits'}), {len(set(d['flop_key']))} distinct canonical flops")
    zs = (d["r_oop"] * d["v_oop"]).sum(1) + (d["r_ip"] * d["v_ip"]).sum(1)
    print(f"label zero-sum check: r.v = {zs.min():.6f} .. {zs.max():.6f}  (must be 1)")
    eq_zs = (d["r_oop"] * d["f_eq_oop"]).sum(1) + (d["r_ip"] * d["f_eq_ip"]).sum(1)
    print(f"equity zero-sum check: {eq_zs.min():.6f} .. {eq_zs.max():.6f}  (must be 1)\n")
    preds = {"const 0.5": evaluate.constant_half(d), "range-avg oracle": evaluate.range_average_oracle(d),
             "EQUITY (floor)": evaluate.equity_baseline(d)}
    print("range-weighted MAE, % of pot (lower is better)")
    print(evaluate.report(d, preds))


if __name__ == "__main__":
    main()
