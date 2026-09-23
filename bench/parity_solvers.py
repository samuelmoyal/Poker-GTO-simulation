#!/usr/bin/env python3
"""Parity of the two solver backends of the trainer: TexasSolver (`gto.solver`) vs postflop-solver (`gto.native`).

Both get the SAME input (class ranges of a preflop matchup, expanded to combos for the native one) and are compared on
the trees they dump: the action sets, and the strategy difference reach-weighted over every node the two trees share.
A noise floor puts the numbers in scale: the native solver against itself at two different accuracies.

  python3 bench/parity_solvers.py --n 6 --threads 4
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gto-trainer"))
from gto import config, native, ranges, solver  # noqa: E402
from gto.cards import DECK, canon  # noqa: E402


def strat_of(node):
    return {canon(k): v for k, v in node["strategy"]["strategy"].items()}


def compare(a, b, w_ip, w_oop):
    """Reach-weighted mean |freq difference| over the nodes present in both trees, plus what does not match."""
    stats = dict(num=0.0, den=0.0, root=None, missing=0, nodes=0)

    def walk(x, y, reach, opp_mass, is_root):
        if x.get("node_type") != "action_node" or y.get("node_type") != "action_node":
            return
        sx, sy = strat_of(x), strat_of(y)
        ax, ay = x["actions"], y["actions"]
        common = [l for l in ax if l in ay]
        stats["missing"] += (len(ax) - len(common)) + (len(ay) - len(common))
        p = x["player"]
        hands = [h for h in sx if h in sy]
        rmass = sum(reach[p].get(h, 0.0) for h in hands) or 1e-12
        diff = 0.0
        for h in hands:
            w = reach[p].get(h, 0.0)
            d = sum(abs(sx[h][ax.index(l)] - sy[h][ay.index(l)]) for l in common) / 2  # total-variation over common actions
            diff += w * d
        diff /= rmass
        node_w = rmass * opp_mass
        stats["num"] += diff * node_w
        stats["den"] += node_w
        stats["nodes"] += 1
        if is_root:
            stats["root"] = diff
        for l in common:
            cx, cy = x.get("childrens", {}).get(l), y.get("childrens", {}).get(l)
            if not cx or not cy:
                continue
            new = [dict(reach[0]), dict(reach[1])]
            i = ax.index(l)
            for h in hands:
                new[p][h] = reach[p].get(h, 0.0) * sx[h][i]
            walk(cx, cy, new, sum(reach[1 - p].values()), False)

    reach0 = [{canon(c): w for c, w in w_ip.items()}, {canon(c): w for c, w in w_oop.items()}]  # 0 = IP (TexasSolver)
    walk(a, b, reach0, sum(reach0[1 - a["player"]].values()), True)
    return dict(root=stats["root"], mean=stats["num"] / max(stats["den"], 1e-12), nodes=stats["nodes"], missing=stats["missing"])


def spots(n, seed):
    rng = random.Random(seed)
    names = list(config.MATCHUPS)
    out = []
    for i in range(n):
        m = names[i % len(names)]
        street = "turn" if i % 2 == 0 else "river"
        cards = rng.sample(DECK, 4 if street == "turn" else 5)
        out.append((m, street, cards))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    config.THREADS = args.threads
    print(f"{'matchup':11} {'street':5} {'board':12} {'TS s':>6} {'nat s':>6} {'Δroot':>7} {'Δmean':>7} {'noise':>7} {'act≠':>5}")
    for m, street, board in spots(args.n, args.seed):
        mm = config.MATCHUPS[m]
        ip_w, oop_w = ranges.load_matchup_ranges(m)
        pot = int(mm["pot_bb"] * config.SCALE * 2)
        stack = int(mm["stack_bb"] * config.SCALE - mm["pot_bb"] * config.SCALE / 2)
        ips, oops = ranges.range_string(ip_w), ranges.range_string(oop_w)
        t = time.time()
        ts = solver.solve(board, pot, stack, ips, oops, street)
        t_ts = time.time() - t
        t = time.time()
        nat = native.solve(board, pot, stack, ips, oops, street)
        t_nat = time.time() - t
        loose = native.solve(board, pot, stack, ips, oops, street, effort=dict(target_pct=1.5), cache_path=os.path.join(config.NATIVE_CACHE_DIR, f"loose_{street}_{''.join(board)}_{m}.json.gz"))
        tight = native.solve(board, pot, stack, ips, oops, street, effort=dict(target_pct=0.1, max_iter=1500), cache_path=os.path.join(config.NATIVE_CACHE_DIR, f"tight_{street}_{''.join(board)}_{m}.json.gz"))
        board_dead = list(board)
        w_ip, w_oop = ranges.combo_weights(ip_w, board_dead), ranges.combo_weights(oop_w, board_dead)
        c = compare(ts, nat, w_ip, w_oop)
        noise = compare(loose, tight, w_ip, w_oop)
        print(f"{m:11} {street:5} {''.join(board):12} {t_ts:6.1f} {t_nat:6.1f} {c['root']:7.3f} {c['mean']:7.3f} {noise['mean']:7.3f} {c['missing']:5d}", flush=True)


if __name__ == "__main__":
    main()
