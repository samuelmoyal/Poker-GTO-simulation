"""Solve turn-root states with the Rust `turn-labels` binary and store them as Parquet (zstd).

Row schema (one row per state; per-hand vectors follow `cards.COMBOS`, 0 where the combo is absent):
  id, split, matchup, kind, line, perturbed, board uint8[4], pot_bb, stack_bb, spr, tree
  in_oop, in_ip      float32[1326]  the ranges as given to the solver, normalised to sum 1 (the NET INPUT)
  r_oop, r_ip        float32[1326]  the solver's card-removal-corrected marginals (sum to 1): c_i(h) ~ in_i(h) * Z_i(h)
  source, flop_key   int8, string   1 TexasSolver flop lib / 2 native flop solve; canonical flop (split key)
  v_oop, v_ip        float32[1326]  EV / pot   (constant-sum: r_oop.v_oop + r_ip.v_ip == 1)
  root_actions       list<string>   OOP's first decision, e.g. [Check, Bet(79), AllIn(940)]
  root_strategy      float32[4*1326] [action][hand], zero padded to 4 actions
  expl_pct, iters, solve_s, ev_sum_err
"""
import glob
import os
import subprocess
import threading
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import cards, paths

MAX_ROOT_ACTIONS = 4
TREE = "turn:66%,a/2.5x river:66%,a/2.5x allin<=1.5 force<=0.15"
SCALE = 10  # chips per bb (matches the trainer)


class SolverCrashed(RuntimeError):
    pass


class LabelSolver:
    """A long-lived `turn-labels` process; one JSON line in, one out."""

    def __init__(self, binary=paths.LABEL_BIN, timeout=300):
        self.binary, self.timeout, self.p = binary, timeout, None
        self._start()

    def _start(self):
        self.p = subprocess.Popen([self.binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def solve(self, request: dict) -> dict:
        import json
        timer = threading.Timer(self.timeout, self.p.kill)
        timer.start()
        try:
            self.p.stdin.write(json.dumps(request) + "\n")
            self.p.stdin.flush()
            line = self.p.stdout.readline()
        except (BrokenPipeError, OSError):
            line = ""
        finally:
            timer.cancel()
        if not line:
            self.close()
            self._start()
            raise SolverCrashed(f"turn-labels died or timed out on {request['id']}")
        return json.loads(line)

    def close(self):
        if self.p and self.p.poll() is None:
            self.p.kill()
        self.p = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _vec(hands, values) -> np.ndarray:
    out = np.zeros(cards.NUM_COMBOS, dtype=np.float32)
    out[[cards.combo_index(h) for h in hands]] = values
    return out


def to_row(state, resp: dict) -> dict:
    pot = state.pot
    root = np.zeros((MAX_ROOT_ACTIONS, cards.NUM_COMBOS), dtype=np.float32)
    n_act = len(resp["root_actions"])
    if n_act > MAX_ROOT_ACTIONS:
        raise ValueError(f"{n_act} root actions")
    strat = np.array(resp["root_strategy"], dtype=np.float32).reshape(n_act, -1)
    idx = [cards.combo_index(h) for h in resp["oop_hands"]]
    root[:n_act, idx] = strat
    r_oop, r_ip = _vec(resp["oop_hands"], resp["oop_w"]), _vec(resp["ip_hands"], resp["ip_w"])
    r_oop, r_ip = r_oop / r_oop.sum(), r_ip / r_ip.sum()  # the solver's "normalized" weights sum to #valid pairs, not 1
    v_oop, v_ip = _vec(resp["oop_hands"], np.array(resp["oop_ev"]) / pot), _vec(resp["ip_hands"], np.array(resp["ip_ev"]) / pot)
    in_oop, in_ip = (_vec(list(d), list(d.values())) for d in (state.oop, state.ip))
    return dict(
        in_oop=in_oop / in_oop.sum(), in_ip=in_ip / in_ip.sum(), source=state.source,
        flop_key=cards.canonical_flop_key(state.board),
        id=state.id, split=state.split, matchup=state.matchup, kind=state.kind, line=state.line,
        perturbed=state.perturbed, board=np.array([cards.CARD_ID[c] for c in state.board], dtype=np.uint8),
        pot_bb=pot / SCALE, stack_bb=state.stack / SCALE, spr=state.stack / pot, tree=TREE,
        r_oop=r_oop, r_ip=r_ip, v_oop=v_oop, v_ip=v_ip, root_actions=resp["root_actions"],
        root_strategy=root.ravel(), expl_pct=resp["expl_pct"], iters=resp["iters"], solve_s=resp["solve_s"],
        ev_sum_err=resp["ev_sum_err"] / pot,
    )


def _fixed(rows, key, width, dtype):
    flat = np.concatenate([np.asarray(r[key], dtype=dtype).ravel() for r in rows])
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), width)


def write_shard(rows, out_dir=paths.TURN_LABELS_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cols = {k: [r[k] for r in rows] for k in ("id", "split", "matchup", "kind", "line", "perturbed", "tree")}
    for k in ("pot_bb", "stack_bb", "spr", "expl_pct", "solve_s", "ev_sum_err"):
        cols[k] = pa.array([r[k] for r in rows], pa.float32())
    cols["iters"] = pa.array([r["iters"] for r in rows], pa.int32())
    cols["source"] = pa.array([r["source"] for r in rows], pa.int8())
    cols["flop_key"] = pa.array([r["flop_key"] for r in rows])
    cols["board"] = _fixed(rows, "board", 4, np.uint8)
    for k in ("in_oop", "in_ip", "r_oop", "r_ip", "v_oop", "v_ip"):
        cols[k] = _fixed(rows, k, cards.NUM_COMBOS, np.float32)
    cols["root_strategy"] = _fixed(rows, "root_strategy", MAX_ROOT_ACTIONS * cards.NUM_COMBOS, np.float32)
    cols["root_actions"] = pa.array([r["root_actions"] for r in rows], pa.list_(pa.string()))
    path = os.path.join(out_dir, f"part-{uuid.uuid4().hex[:8]}.parquet")
    pq.write_table(pa.table(cols), path, compression="zstd")
    return path


def existing_ids(out_dir=paths.TURN_LABELS_DIR) -> set:
    ids = set()
    for p in glob.glob(os.path.join(out_dir, "part-*.parquet")):
        ids.update(pq.read_table(p, columns=["id"]).column("id").to_pylist())
    return ids
