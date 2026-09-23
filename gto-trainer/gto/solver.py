"""Street solver: board, pot, stack, ip/oop range, street -> the solved betting tree of that street, cached on disk.

The work is done by `tools/turn-labels`' `street_tree` binary (postflop-solver).  Ranges may be given per combo
(`{"AhKd": 0.4}`) or as a class string (`"AKs:0.4,QQ"`).  The tree comes back in the layout `engine.py` reads:
action nodes with `actions`, `strategy.strategy[combo]` and `childrens`; player 0 is IP.
"""
import gzip
import hashlib
import json
import os
import subprocess
import threading

from . import config, ranges


class SolverError(RuntimeError):
    pass


_run_lock = threading.Lock()  # one solve at a time: the solver already uses every core


def _read_gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _write_gz(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def _sizes(cfg) -> dict:
    """{bet, raise, donk} strings: explicit all-ins are listed, the solver's automatic ones are switched off."""
    bet = [f"{s:g}%" for s in cfg["bet"]] + (["a"] if cfg["allin"] else [])
    raise_ = [f"{s:g}%" for s in cfg["raise_"]] + (["a"] if cfg["allin"] else [])
    return dict(bet=", ".join(bet), raise_=", ".join(raise_), donk=", ".join(f"{s:g}%" for s in cfg["donk"]))


def _combos(rng, board):
    if isinstance(rng, str):
        rng = ranges.combo_weights(ranges.parse_range(rng), board)
    return {c: round(float(w), 5) for c, w in rng.items() if w > 0}


def build_request(board, pot, stack, ip_range, oop_range, street, effort=None, progress=False) -> dict:
    tree = config.BET_TREE[street]
    req = dict(
        board="".join(board), pot=int(pot), stack=int(stack),
        oop=_combos(oop_range, board), ip=_combos(ip_range, board),
        add_allin=0.0, force_allin=0.15, progress=progress,
        **dict(config.EFFORT[street], **(effort or {})),
    )
    for st, cfg in tree.items():
        s = _sizes(cfg)
        req[st] = {"bet": s["bet"], "raise": s["raise_"], "donk": s["donk"]}
    return req


def solve(board, pot, stack, ip_range, oop_range, street, cache_path=None, on_progress=None, effort=None):
    """Solve one street and return the tree (dict). Cached on disk by request hash."""
    req = build_request(board, pot, stack, ip_range, oop_range, street, effort, progress=on_progress is not None)
    if cache_path is None:
        blob = json.dumps({k: v for k, v in req.items() if k != "progress"}, sort_keys=True)
        cache_path = os.path.join(config.STREET_CACHE_DIR, f"{street}_{hashlib.sha1(blob.encode()).hexdigest()}.json.gz")
    if os.path.exists(cache_path):
        try:
            return _read_gz(cache_path)
        except (OSError, EOFError, json.JSONDecodeError):
            os.remove(cache_path)

    with _run_lock:
        if os.path.exists(cache_path):
            return _read_gz(cache_path)
        if not os.path.exists(config.SOLVER_BIN):
            raise SolverError(f"{config.SOLVER_BIN} manquant : cd tools/turn-labels && cargo build --release")
        req["id"] = "r"
        env = dict(os.environ, RAYON_NUM_THREADS=str(config.THREADS))
        proc = subprocess.Popen([config.SOLVER_BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=env, start_new_session=True)
        try:
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            proc.stdin.close()
            resp = None
            for line in proc.stdout:
                msg = json.loads(line)
                if "progress" in msg:
                    if on_progress:
                        on_progress(dict(iter=msg["progress"]["iter"], exploitability=msg["progress"]["expl_pct"]))
                else:
                    resp = msg
        finally:
            proc.wait()
        if resp is None or not resp.get("ok"):
            err = (resp or {}).get("error") or proc.stderr.read()[-400:]
            raise SolverError(f"Solveur natif en échec ({street} {board}): {err}")
        tree = resp["tree"]
        _write_gz(cache_path, tree)
        with open(cache_path.replace(".json.gz", ".meta.json"), "w") as f:
            json.dump(dict(iter=resp["iters"], exploitability=resp["expl_pct"], seconds=round(resp["solve_s"], 2),
                           street=street, board=list(board)), f)
        return tree
