#!/usr/bin/env python3
"""Validate the resolver's maths WITHOUT a network: a turn resolve whose river leaves are solved exactly by
postflop-solver must reproduce the solver's own turn-root solve (same trees: bet 66%, no raise, no all-in).

  model/.venv/bin/python bench/validate_resolver.py [--iters 100 --refresh 5]
Runs the resolve with and without the eps-floor on absent hands to show what the floor is for.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import numpy as np  # noqa: E402
from gtonet import cards, cfr, labels, oracle, states  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--refresh", type=int, default=5)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    st = next(s for s in states.flop_states(states.flop_files()[0], 12, seed=args.seed) if not s.perturbed)
    P, S = 100.0, 400.0
    r0, r1 = (labels._vec(list(d), list(d.values())) for d in (st.oop, st.ip))
    with labels.LabelSolver(timeout=600) as solver:
        truth = solver.solve({"id": "t", "flop": "".join(st.board[:3]), "turn": st.board[3], "pot": int(P), "stack": int(S),
                              "oop": st.oop, "ip": st.ip, "target_pct": 0.02, "max_iter": 4000, "bets": "66%", "raise": "",
                              "add_allin": 0.0, "force_allin": 0.0})
        assert truth["ok"], truth
        print(f"board {' '.join(st.board)} | hands {len(st.oop)}/{len(st.ip)} | solver turn-root: {truth['iters']} it, "
              f"{truth['solve_s']:.2f}s, exploitability {truth['expl_pct']:.3f}%")
        w = {s: np.array(truth[f"{s}_w"]) / np.sum(truth[f"{s}_w"]) for s in ("oop", "ip")}
        bet_true = float((w["oop"] * np.array(truth["root_strategy"]).reshape(len(truth["root_actions"]), -1)[1]).sum())
        for floor in (0.0, 1e-3):
            root = cfr.build_round_tree(P, S, bet_fracs=(0.66,), raise_fracs=(), max_raises=0)
            leaves = oracle.ExactRiverLeaves(st.board, P, S, solver, (r0, r1), floor)
            game = cfr.Cfr(root, r0, r1, P, leaves)
            t0 = time.time()
            for it in range(args.iters):
                if it % args.refresh == 0:
                    leaves.refresh(game.end_reaches())
                game.iterate()
            ev = game.ev_per_hand()
            errs = []
            for side, mine in zip(("oop", "ip"), ev):
                idx = [cards.combo_index(h) for h in truth[f"{side}_hands"]]
                e = np.abs(mine[idx] - np.array(truth[f"{side}_ev"]))
                errs.append(100 * float((w[side] * e).sum() / P))
            s_root = game.avg_sigma(root)
            wr = r0 * cfr.zmass(r1)
            wr = wr / wr.sum()
            print(f"floor {floor:g}: EV error oop {errs[0]:.2f}% / ip {errs[1]:.2f}% of pot | root bet freq {float((wr * s_root[1]).sum()):.3f} "
                  f"(solver {bet_true:.3f}) | {leaves.n_solves} river solves, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
