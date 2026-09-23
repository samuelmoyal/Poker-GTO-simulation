#!/usr/bin/env python3
"""Label shards (schema v2) -> featurised dataset shards. Resumable: shards already done are skipped."""
import glob
import os
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import dataset, paths  # noqa: E402


def _one(p):
    t0 = time.time()
    out = dataset.build_shard(p)
    n = pq.read_metadata(out).num_rows
    return f"{os.path.basename(out)}: {n} rows in {time.time() - t0:.0f}s ({(time.time() - t0) / n:.2f} s/row)"


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    todo = []
    for p in sorted(glob.glob(os.path.join(paths.TURN_LABELS_DIR, "part-*.parquet"))):
        if "in_oop" not in pq.read_schema(p).names:
            print("skip (run backfill_v2.py first):", os.path.basename(p))
        elif not os.path.exists(os.path.join(paths.DATASET_DIR, "feat-" + os.path.basename(p))):
            todo.append(p)
    print(f"{len(todo)} shard(s) to featurise", flush=True)
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            for line in ex.map(_one, todo):
                print(line, flush=True)
    else:
        for p in todo:
            print(_one(p), flush=True)


if __name__ == "__main__":
    main()
