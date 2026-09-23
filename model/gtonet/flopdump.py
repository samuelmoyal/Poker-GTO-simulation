"""Replay a TexasSolver flop solve into the per-combo ranges that reach the turn.

Each `chance_node` of the flop tree (check-check, bet-call, ...) is the start of the turn.  Multiplying the
preflop weights by the solver's action frequencies along the path gives both players' exact combo ranges,
which is what the value net is trained on (the TexasSolver CLI itself only accepts class-level weights).
"""
import gzip
import json
import os
from dataclasses import dataclass

from . import paths  # noqa: F401
from gto import config, preflop, ranges  # noqa: E402

IP, OOP = 0, 1  # player ids in the solver JSON


@dataclass
class FlopLine:
    line: str        # e.g. "x-b28-c"
    pot: int         # chips (SCALE per bb) when the turn is dealt
    stack: int       # effective stack behind
    reach: tuple     # (ip {hand: reach}, oop {hand: reach})
    mass: float      # share of both ranges that gets here (blockers ignored)


def parse_name(path: str):
    matchup, board, pid = os.path.basename(path)[: -len(".json.gz")].split("__")
    return matchup, [board[i:i + 2] for i in range(0, 6, 2)], pid


def load(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def root_reach(matchup: str, board):
    """Preflop weights as the solver received them (4-decimal class weights), per unblocked combo."""
    ip_w, oop_w = ranges.load_matchup_ranges(matchup)
    return tuple(ranges.combo_weights({h: round(x, 4) for h, x in cw.items()}, dead=board) for cw in (ip_w, oop_w))


def _label(kind: str, amount: str) -> str:
    return {"CHECK": "x", "CALL": "c"}.get(kind) or f"{'b' if kind == 'BET' else 'r'}{int(float(amount))}"


def replay(tree: dict, matchup: str, board) -> list:
    m = preflop.get(matchup)
    pot0, stack0 = round(m["pot_bb"] * config.SCALE), round(m["stack_bb"] * config.SCALE)
    root = root_reach(matchup, board)
    totals = [sum(r.values()) for r in root]
    first = set(tree["strategy"]["strategy"])
    if first != set(root[OOP]):
        raise ValueError(f"{matchup} {board}: solver hands != preflop combos ({len(first)} vs {len(root[OOP])})")
    out = []

    def rec(node, reach, inv, line):
        if node["node_type"] == "chance_node":
            mass = (sum(reach[IP].values()) / totals[IP]) * (sum(reach[OOP].values()) / totals[OOP])
            pot, stack = round(pot0 + inv[IP] + inv[OOP]), round(stack0 - max(inv))
            out.append(FlopLine("-".join(line), pot, stack, tuple(reach), mass))
            return
        p = node["player"]
        strat = node["strategy"]["strategy"]
        for k, action in enumerate(node["strategy"]["actions"]):
            kind, _, amount = action.partition(" ")
            if kind == "FOLD":
                continue
            if kind not in ("CHECK", "CALL", "BET", "RAISE"):
                raise ValueError(f"unsupported action {action!r}")
            new_reach = list(reach)
            new_reach[p] = {h: reach[p][h] * pr[k] for h, pr in strat.items() if h in reach[p] and pr[k] > 0}
            inv2 = list(inv)
            if kind in ("BET", "RAISE"):
                inv2[p] = float(amount)  # RAISE x = raise *to* x
            elif kind == "CALL":
                inv2[p] = inv[1 - p]
            rec(node["childrens"][action], new_reach, inv2, line + [_label(kind, amount)])

    rec(tree, list(root), [0.0, 0.0], [])
    return out
