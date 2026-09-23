"""Heads-up postflop spots read off the solver's preflop range tree (ranges/6max_range).

A tree path such as `BTN/2.5bb/BB/11.0bb/BTN/Call` is a list of (player, action) pairs: BTN opens 2.5bb,
BB 3-bets to 11bb, BTN calls.  Every `Call` folder whose two live players both have a range file is a
heads-up postflop spot; players who never appear in the path (or fold) leave dead money in the pot.
"""
import os
from functools import lru_cache

from . import config

ORDER = ["SB", "BB", "UTG", "MP", "CO", "BTN"]  # postflop acting order: first = out of position
STACK_BB = 100.0
KINDS = {1: "srp", 2: "3bet", 3: "4bet"}
_VERBS = {1: "ouvre", 2: "3-bet", 3: "4-bet"}


def _replay(path: str):
    """-> (chips each player put in, live players, number of raises, labels)."""
    parts = path.split("/")
    contrib = {"SB": 0.5, "BB": 1.0}
    cur, raises, last, words = 1.0, 0, {}, []
    for pos, act in zip(parts[0::2], parts[1::2]):
        last[pos] = act
        if act == "Call":
            contrib[pos] = cur
            words.append(f"{pos} call")
        elif act == "Fold":
            words.append(f"{pos} fold")
        else:
            raises += 1
            cur = STACK_BB if act == "AllIn" else float(act[:-2])
            contrib[pos] = cur
            words.append(f"{pos} " + ("all-in" if act == "AllIn" else _VERBS.get(raises, f"{raises + 1}-bet")))
    live = [p for p, a in last.items() if a != "Fold"]
    return contrib, live, raises, words


@lru_cache(maxsize=1)
def lines() -> dict:
    """{key: matchup dict} for every heads-up preflop line that reaches the flop with chips behind.

    Same schema as `config.MATCHUPS` plus `kind` (srp / 3bet / 4bet).  The eight hand-picked SRP
    matchups keep their historical keys, so their cached flops stay valid.
    """
    legacy = {m["dir"]: k for k, m in config.MATCHUPS.items()}
    out = {}
    for d, _, files in os.walk(config.RANGES_DIR):
        path = os.path.relpath(d, config.RANGES_DIR)
        if os.path.basename(d) != "Call":
            continue
        contrib, live, raises, words = _replay(path)
        have = {f[: -len("_range.txt")] for f in files if f.endswith("_range.txt")}
        if len(live) != 2 or not set(live) <= have:
            continue
        stack = STACK_BB - max(contrib[p] for p in live)
        if stack <= 0:  # all-in and called: no postflop decisions
            continue
        oop, ip = sorted(live, key=ORDER.index)
        key = legacy.get(path) or path.replace("bb", "").replace("/", "-")  # no "__": it separates cache fields
        out[key] = dict(label=", ".join(words), ip=ip, oop=oop, dir=path, pot_bb=sum(contrib.values()),
                        stack_bb=stack, kind=KINDS.get(raises, f"{raises + 1}bet"))
    return dict(sorted(out.items()))


def get(key: str) -> dict:
    """Matchup by key: the app's SRP matchups first, then any other line of the preflop tree."""
    m = config.MATCHUPS.get(key)
    return dict(m, kind="srp") if m else lines()[key]
