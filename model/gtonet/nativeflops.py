"""Flop solves by the native solver (`flop_lines`): the same per-line turn ranges as `flopdump.replay`, ~5x faster.

Stored as `data/native_flops/<matchup>__<flop>__native.json.gz` (+ `.meta.json` with the exploitability), so
`states.flop_files()` treats them like the cached TexasSolver flops.
"""
import gzip
import json
import os

from . import flopdump, labels, paths

PID = "native"
TREE = "flop:50%+raise100% turn/river:66% no-raise, force_allin<=0.15"


def path_for(matchup: str, flop) -> str:
    return os.path.join(paths.NATIVE_FLOPS, f"{matchup}__{''.join(flop)}__{PID}.json.gz")


def solve_flop(solver: labels.LabelSolver, matchup: str, flop, target_pct=0.5) -> dict:
    """Solve one flop; writes the cache files and returns the response."""
    from gto import config, preflop  # noqa: E402
    m = preflop.get(matchup)
    ip, oop = flopdump.root_reach(matchup, flop)
    resp = solver.solve(dict(
        id=f"{matchup}__{''.join(flop)}", flop="".join(flop), pot=round(m["pot_bb"] * config.SCALE),
        stack=round(m["stack_bb"] * config.SCALE), oop=oop, ip=ip, target_pct=target_pct))
    if not resp["ok"]:
        return resp
    path = path_for(matchup, flop)
    os.makedirs(paths.NATIVE_FLOPS, exist_ok=True)
    with gzip.open(path + ".tmp", "wt") as f:
        json.dump(dict(matchup=matchup, flop=list(flop), tree=TREE, lines=resp["lines"]), f, separators=(",", ":"))
    os.replace(path + ".tmp", path)
    meta = {k: resp[k] for k in ("iters", "expl_pct", "solve_s", "mem_gb")}
    meta["exploitability"] = meta["expl_pct"]  # same key as the TexasSolver sidecars
    with open(path.replace(".json.gz", ".meta.json"), "w") as f:
        json.dump(meta, f)
    return resp


def load_lines(path: str) -> list:
    """FlopLine list from a native file (mass computed against the preflop root ranges, like `replay`)."""
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    root = flopdump.root_reach(d["matchup"], d["flop"])
    totals = [sum(r.values()) for r in root]
    out = []
    for l in d["lines"]:
        reach = (l["ip"], l["oop"])
        mass = (sum(reach[0].values()) / totals[0]) * (sum(reach[1].values()) / totals[1])
        out.append(flopdump.FlopLine(l["line"], l["pot"], l["stack"], reach, mass))
    return out
