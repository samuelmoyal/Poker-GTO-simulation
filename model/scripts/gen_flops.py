#!/usr/bin/env python3
"""Solve random flops with the native solver (resumable): the ranges that reach the turn feed gen_labels.py.

  .venv/bin/python scripts/gen_flops.py --total 300                   # pot types 50/35/15 over the preflop tree
  .venv/bin/python scripts/gen_flops.py --total 50 --kind 3bet 4bet
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import labels, nativeflops, paths  # noqa: E402
from gto import preflop, spots  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--mix", default="srp=0.5,3bet=0.35,4bet=0.15", help="pot-type weights")
    ap.add_argument("--kind", nargs="+", help="restrict to these pot types")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-pct", type=float, default=0.5)
    args = ap.parse_args()

    pool = {}
    for key, m in preflop.lines().items():
        if not args.kind or m["kind"] in args.kind:
            pool.setdefault(m["kind"], []).append(key)
    mix = {k: float(v) for k, v in (t.split("=") for t in args.mix.split(","))}
    kinds = [k for k in mix if pool.get(k)]
    rng = random.Random(args.seed)
    jobs = set()
    while len(jobs) < args.total:
        kind = rng.choices(kinds, [mix[k] for k in kinds])[0]
        jobs.add((rng.choice(pool[kind]), tuple(spots.random_canonical_flop(rng))))
    jobs = sorted(jobs)
    rng.shuffle(jobs)
    print(f"{len(jobs)} flops, tree: {nativeflops.TREE}", flush=True)

    t0, done, failed = time.time(), 0, 0
    with labels.LabelSolver(paths.FLOP_BIN, timeout=1800) as solver:
        for k, (matchup, flop) in enumerate(jobs, 1):
            tag = f"[{k}/{len(jobs)}] {preflop.get(matchup)['kind']} {matchup} {''.join(flop)}"
            if os.path.exists(nativeflops.path_for(matchup, flop)):
                print(tag, "already solved", flush=True)
                continue
            try:
                r = nativeflops.solve_flop(solver, matchup, flop, args.target_pct)
            except labels.SolverCrashed as e:
                failed += 1
                print(tag, "CRASH", e, flush=True)
                continue
            if not r["ok"]:
                failed += 1
                print(tag, "FAIL", r.get("error"), flush=True)
                continue
            done += 1
            print(f"{tag} {r['solve_s']:.0f}s, {r['iters']} it, {r['expl_pct']:.3f}%, {r['mem_gb']:.1f} GB, "
                  f"{len(r['lines'])} lines", flush=True)
    print(f"done: {done} solved, {failed} failed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
