#!/usr/bin/env python3
"""Add the schema-v2 columns (raw input ranges, source, flop_key) to shards written before they existed.

States are deterministic (seed 0, 24 per flop), so the id of every old row maps back to its TurnState.
Idempotent; run it again after any generation that used the old code.
"""
import glob
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import cards, labels, paths, states  # noqa: E402


def main(seed=0, n_per_flop=24):
    todo = [p for p in sorted(glob.glob(os.path.join(paths.TURN_LABELS_DIR, "part-*.parquet")))
            if "in_oop" not in pq.read_schema(p).names]
    print(f"{len(todo)} shard(s) to backfill")
    if not todo:
        return
    files = glob.glob(os.path.join(paths.FLOP_CACHE, "*.json.gz")) + glob.glob(os.path.join(paths.NATIVE_FLOPS, "*.json.gz"))
    by_id = {}
    for f in files:
        try:
            for s in states.flop_states(f, n_per_flop, seed):
                by_id[s.id] = s
        except (KeyError, ValueError):
            continue
    for p in todo:
        t = pq.read_table(p)
        sts = [by_id.get(i) for i in t.column("id").to_pylist()]
        if any(s is None for s in sts):
            print(f"  {os.path.basename(p)}: {sum(s is None for s in sts)} unknown ids, left untouched")
            continue
        cols = {}
        for k, attr in (("in_oop", "oop"), ("in_ip", "ip")):
            vecs = []
            for s in sts:
                d = getattr(s, attr)
                v = labels._vec(list(d), list(d.values()))
                vecs.append(v / v.sum())
            cols[k] = pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(vecs)), cards.NUM_COMBOS)
        for name, arr in (("in_oop", cols["in_oop"]), ("in_ip", cols["in_ip"]),
                          ("source", pa.array([s.source for s in sts], pa.int8())),
                          ("flop_key", pa.array([cards.canonical_flop_key(s.board) for s in sts]))):
            t = t.append_column(name, arr)
        pq.write_table(t, p + ".tmp", compression="zstd")
        os.replace(p + ".tmp", p)
        print(f"  {os.path.basename(p)}: {t.num_rows} rows backfilled")


if __name__ == "__main__":
    main()
