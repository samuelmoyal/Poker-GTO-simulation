"""Featurised turn-root dataset (Parquet) -> numpy, and the exact features the equity baseline / net consume.

`build_shard` adds to a label shard the per-hand features of architecture.md §8.4 (all exact, none learned):
  f_eq_unif  equity vs a uniform range on this board (hand-strength rank)      float32[1326]
  f_eq_oop   equity of each hand held by OOP vs the IP input range             float32[1326]
  f_eq_ip    equity of each hand held by IP vs the OOP input range             float32[1326]
  f_cat      made-hand category on the turn (0..8)                             int8[1326]
  f_outs     river cards that raise the category                               int8[1326]
"""
import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import cards, equity, paths

FEATURES = ("f_eq_unif", "f_eq_oop", "f_eq_ip", "f_cat", "f_outs")


def _col(table, name):
    col = table.column(name).combine_chunks()
    if pa.types.is_fixed_size_list(col.type):
        return col.values.to_numpy(zero_copy_only=False).reshape(len(col), -1)
    return np.array(col.to_pylist()) if pa.types.is_string(col.type) else col.to_numpy(zero_copy_only=False)


def build_shard(path: str, out_dir=paths.DATASET_DIR, cache_size=64) -> str:
    out = os.path.join(out_dir, "feat-" + os.path.basename(path))
    os.makedirs(out_dir, exist_ok=True)
    t = pq.read_table(path)
    board, in_oop, in_ip = _col(t, "board"), _col(t, "in_oop"), _col(t, "in_ip")
    feats = {k: [] for k in FEATURES}
    tables = {}
    import torch
    for i in range(t.num_rows):
        key = tuple(board[i])
        if key not in tables:
            if len(tables) >= cache_size:
                tables.clear()
            tables[key] = equity.BoardTables(list(key))
        tb = tables[key]
        r = torch.tensor(np.stack([in_oop[i], in_ip[i]]), dtype=torch.float32, device=tb.device)
        f = dict(f_eq_unif=tb.uniform_equity().cpu().numpy(), f_eq_oop=tb.equity(r[1]).cpu().numpy(),
                 f_eq_ip=tb.equity(r[0]).cpu().numpy(), f_cat=tb.cat6, f_outs=tb.outs)
        for k in FEATURES:
            feats[k].append(f[k])
    for k in FEATURES:
        dtype = np.int8 if k in ("f_cat", "f_outs") else np.float32
        flat = np.concatenate(feats[k]).astype(dtype)
        t = t.append_column(k, pa.FixedSizeListArray.from_arrays(pa.array(flat), cards.NUM_COMBOS))
    pq.write_table(t, out + ".tmp", compression="zstd")
    os.replace(out + ".tmp", out)
    return out


def load(split=None, out_dir=paths.DATASET_DIR) -> dict:
    """All featurised rows (optionally one split) as a dict of numpy arrays / lists."""
    tables = [pq.read_table(p) for p in sorted(glob.glob(os.path.join(out_dir, "feat-part-*.parquet")))]
    if not tables:
        raise FileNotFoundError(f"no featurised shards in {out_dir}; run scripts/featurize.py")
    cols = tables[0].column_names  # backfilled and freshly written shards hold the same columns in a different order
    t = pa.concat_tables([x.select(cols) for x in tables])
    if split:
        t = t.filter(pa.compute.equal(t.column("split"), split))
    d = {name: _col(t, name) for name in t.column_names if name != "root_actions"}
    d["root_actions"] = t.column("root_actions").to_pylist()
    return d


def texture(board) -> str:
    """Flop texture of a 4-card board (ids): monotone / twotone / rainbow, plus paired."""
    flop = [int(c) for c in board[:3]]
    suits, ranks = {c % 4 for c in flop}, [c // 4 for c in flop]
    kind = {1: "monotone", 2: "twotone", 3: "rainbow"}[len(suits)]
    return kind + ("+paired" if len(set(ranks)) < 3 else "")
