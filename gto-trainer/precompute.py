#!/usr/bin/env python3
"""Fill the flop library.  Safe to Ctrl-C and resume (solved flops are skipped).

  python3 precompute.py --list                       # every heads-up line of the preflop tree
  python3 precompute.py --total 300                  # 300 random flops, spread over all lines (see --mix)
  python3 precompute.py --total 100 --kind 3bet 4bet
  python3 precompute.py -m BTN_vs_BB -n 20           # 20 random flops for one matchup
  python3 precompute.py -m BTN_vs_BB -f AsKd7c Qh8h3d
  python3 precompute.py                              # 5 random flops for each of the app's SRP matchups
  python3 precompute.py --redo 770851                # re-solve every flop made under an older profile id
GTO_THREADS=4 (env) limits the solver's threads; `nice -n 10` keeps the machine responsive meanwhile.
"""
import argparse
import json
import os
import random
import sys
import time

from gto import config, preflop, solver, spots
from gto.cards import canonical_flop

DEFAULT_MIX = "srp=0.5,3bet=0.35,4bet=0.15"


def parse_mix(text):
    mix = {}
    for tok in text.split(","):
        kind, _, w = tok.partition("=")
        mix[kind.strip()] = float(w)
    return mix


def sample_jobs(total, mix, pool):
    """`total` distinct (matchup, flop): pick a pot type by `mix`, a line of that type uniformly, a random flop."""
    kinds = [k for k in mix if pool.get(k)]
    if not kinds:
        sys.exit("aucune ligne pour ces types de pot")
    jobs = set()
    while len(jobs) < total:
        kind = random.choices(kinds, [mix[k] for k in kinds])[0]
        jobs.add((random.choice(pool[kind]), tuple(spots.random_canonical_flop())))
    return list(jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--matchup", action="append", help="matchup key (repeatable)")
    ap.add_argument("-n", type=int, default=5, help="random flops per matchup")
    ap.add_argument("-f", "--flop", nargs="*", help="explicit flops like AsKd7c")
    ap.add_argument("--kind", nargs="+", choices=sorted(set(preflop.KINDS.values())), help="restrict to these pot types")
    ap.add_argument("--total", type=int, help="sample this many (matchup, flop) jobs across the whole tree")
    ap.add_argument("--mix", default=DEFAULT_MIX, help=f"pot-type weights for --total (default {DEFAULT_MIX})")
    ap.add_argument("--redo", metavar="PROFILE", help="re-solve every flop of the library made under this older profile id")
    ap.add_argument("--list", action="store_true", help="print the available lines and exit")
    args = ap.parse_args()

    lines = preflop.lines()
    pool = {}
    for key, m in lines.items():
        if not args.kind or m["kind"] in args.kind:
            pool.setdefault(m["kind"], []).append(key)

    if args.list:
        for key, m in lines.items():
            print(f"{m['kind']:5} {key:45} pot {m['pot_bb']:6.1f}bb stack {m['stack_bb']:5.1f}bb  {m['label']}")
        print({k: len(v) for k, v in pool.items()}, "lines")
        return

    if args.redo:
        jobs = []
        for name in sorted(os.listdir(config.FLOP_CACHE_DIR)):
            if name.endswith(".json.gz") and name[: -len(".json.gz")].split("__")[2] == args.redo:
                mk, board, _ = name[: -len(".json.gz")].split("__")
                if mk in lines:
                    jobs.append((mk, tuple(board[i:i + 2] for i in range(0, 6, 2))))
    elif args.total:
        jobs = sample_jobs(args.total, parse_mix(args.mix), pool)
    else:
        matchups = args.matchup or ([k for ks in pool.values() for k in ks] if args.kind else list(config.MATCHUPS))
        jobs = []
        for mk in matchups:
            if mk not in lines:
                sys.exit(f"unknown matchup {mk}; see --list")
            if args.flop:
                boards = [canonical_flop([f[i:i + 2] for i in range(0, 6, 2)])[0] for f in args.flop]
            else:
                boards = [spots.random_canonical_flop() for _ in range(args.n)]
            jobs += [(mk, tuple(b)) for b in boards]
    random.shuffle(jobs)
    print(f"{len(jobs)} jobs, profile {config.profile_id()}, flop effort {config.EFFORT['flop']}", flush=True)

    for k, (mk, board) in enumerate(jobs, 1):
        path = spots.flop_path(mk, board)
        tag = f"[{k}/{len(jobs)}] {lines[mk]['kind']} {mk} {''.join(board)}"
        if os.path.exists(path):
            print(tag, "already cached")
            continue
        t0 = time.time()
        try:
            spots.solve_flop(mk, board)
        except solver.SolverError as e:  # e.g. timeout: skip this flop, keep the batch going
            print(f"{tag} FAILED: {e}", flush=True)
            continue
        with open(path.replace(".json.gz", ".meta.json")) as f:
            meta = json.load(f)
        print(f"{tag} solved in {time.time() - t0:.0f}s, {meta.get('iter')} it, exploitability {meta.get('exploitability')}%", flush=True)


if __name__ == "__main__":
    main()
