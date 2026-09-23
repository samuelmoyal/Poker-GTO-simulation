#!/usr/bin/env python3
"""Generate turn-root labels (resumable: states already in the output dir are skipped).

  .venv/bin/python scripts/gen_labels.py --n-per-flop 24            # every cached flop
  .venv/bin/python scripts/gen_labels.py --max-states 200 --out /tmp/x   # small trial

Train states are solved to --target-pct (0.5% of pot); val/test states to --eval-target-pct (0.1%), so the
held-out labels are tighter than the training ones and measure the net's real error, not the solver's noise.
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import labels, paths, states  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-flop", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-states", type=int)
    ap.add_argument("--max-flops", type=int)
    ap.add_argument("--target-pct", type=float, default=0.5)
    ap.add_argument("--eval-target-pct", type=float, default=0.1)
    ap.add_argument("--shard", type=int, default=200, help="rows per Parquet shard")
    ap.add_argument("--out", default=paths.TURN_LABELS_DIR)
    args = ap.parse_args()

    files = states.flop_files()
    random.Random(args.seed).shuffle(files)
    files = files[: args.max_flops]
    done = labels.existing_ids(args.out)
    print(f"{len(files)} flop files, {len(done)} states already labelled", flush=True)

    buf, n, failed, t0 = [], 0, 0, time.time()

    def flush():
        nonlocal buf
        if buf:
            print("wrote", labels.write_shard(buf, args.out), f"({len(buf)} rows)", flush=True)
            buf = []

    try:
        with labels.LabelSolver() as solver:
            for path in files:
                for st in states.flop_states(path, args.n_per_flop, args.seed):
                    if st.id in done:
                        continue
                    target = args.target_pct if st.split == "train" else args.eval_target_pct
                    try:
                        resp = solver.solve(st.request(target))
                    except labels.SolverCrashed as e:
                        failed += 1
                        print("CRASH", e, flush=True)
                        continue
                    if not resp["ok"]:
                        failed += 1
                        print("FAIL", st.id, resp.get("error"), flush=True)
                        continue
                    if abs(resp["ev_sum_err"]) > 1e-4 * st.pot:
                        failed += 1
                        print("NOT ZERO-SUM", st.id, resp["ev_sum_err"], flush=True)
                        continue
                    buf.append(labels.to_row(st, resp))
                    n += 1
                    if n % 25 == 0:
                        print(f"{n} states, {n / (time.time() - t0):.2f}/s, {failed} failed", flush=True)
                    if len(buf) >= args.shard:
                        flush()
                    if args.max_states and n >= args.max_states:
                        return
    finally:
        flush()
        print(f"done: {n} new states, {failed} failed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
