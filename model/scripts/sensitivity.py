#!/usr/bin/env python3
"""Range-sensitivity test against the solver: replace the opponent's range by a linear / polarised one, solve both
exactly, and compare the net's answer AND its shift with the solver's.

  .venv/bin/python scripts/sensitivity.py --n 16 [--split val] [--ckpt checkpoints/net_turn_small.pt]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import cards, dataset, labels, model as M, sensitivity  # noqa: E402


def solve_variant(solver, d, i, in_oop, in_ip):
    def rng(v):
        top = v.max()
        return {cards.combo_str(k): float(v[k] / top) for k in np.flatnonzero(v > 1e-6 * top)}
    board = [cards.DECK[int(c)] for c in d["board"][i]]
    pot = round(float(d["pot_bb"][i]) * labels.SCALE)
    resp = solver.solve(dict(id=f"sens{i}", flop="".join(board[:3]), turn=board[3], pot=pot,
                             stack=round(float(d["stack_bb"][i]) * labels.SCALE), oop=rng(in_oop), ip=rng(in_ip), target_pct=0.1))
    if not resp["ok"]:
        return None
    v = {}
    for side in ("oop", "ip"):
        v[side] = labels._vec(resp[f"{side}_hands"], np.array(resp[f"{side}_ev"]) / pot)
        v["r_" + side] = labels._vec(resp[f"{side}_hands"], resp[f"{side}_w"])
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "net_turn_small.pt"))
    args = ap.parse_args()
    dev = torch.device("cpu")  # small nets, a handful of forwards
    ck = torch.load(args.ckpt, map_location=dev)
    net = M.NetTurn(M.CONFIGS[ck["config"]])
    net.load_state_dict({k.replace("module.", ""): v for k, v in ck["state"].items() if k != "n_averaged"})
    d = dataset.load(args.split)
    idx = np.linspace(0, len(d["id"]) - 1, args.n).astype(int)
    rows = []
    with labels.LabelSolver() as solver:
        for i in idx:
            in_oop, in_ip = np.array(d["in_oop"][i]), np.array(d["in_ip"][i])
            for side in ("oop", "ip"):
                opp = in_ip if side == "oop" else in_oop
                lin, pol = sensitivity.swapped_ranges(np.array(d["f_eq_unif"][i]), opp)
                res = {}
                for name, r in (("lin", lin), ("pol", pol)):
                    a, b = (in_oop, r) if side == "oop" else (r, in_ip)
                    truth = solve_variant(solver, d, i, a, b)
                    if truth is None:
                        break
                    pred = sensitivity.predict_variant(net, d["board"][i], d["spr"][i], a, b, dev)
                    w = truth["r_" + side] / truth["r_" + side].sum()  # the solver's card-removal-corrected marginals
                    res[name] = dict(truth=truth[side], net=pred[side], eq=pred["eq_" + side], w=w)
                if len(res) < 2:
                    continue
                w = res["lin"]["w"]
                shift = lambda k: 100 * float((w * np.abs(res["lin"][k] - res["pol"][k])).sum())  # noqa: E731
                mae = lambda k: 100 * np.mean([(x["w"] * np.abs(x[k] - x["truth"])).sum() for x in res.values()])  # noqa: E731
                rows.append((shift("truth"), shift("net"), shift("eq"), mae("net"), mae("eq")))
    r = np.array(rows)
    print(f"{len(idx)} states x 2 sides, opponent range swapped linear <-> polarised ({args.split}, {os.path.basename(args.ckpt)})")
    print(f"solver shift  {r[:, 0].mean():6.2f}% of pot   (how much the truth moves)")
    print(f"net shift     {r[:, 1].mean():6.2f}%          (corr with solver per case: {np.corrcoef(r[:, 0], r[:, 1])[0, 1]:.2f})")
    print(f"equity shift  {r[:, 2].mean():6.2f}%          (corr {np.corrcoef(r[:, 0], r[:, 2])[0, 1]:.2f})")
    print(f"MAE vs solver on these atypical ranges:  net {r[:, 3].mean():.2f}%   equity floor {r[:, 4].mean():.2f}%")


if __name__ == "__main__":
    main()
