"""Preflop range files -> hand-class weights -> combos, and range strings for the solver."""
import os
import random

from . import config, preflop
from .cards import RANKS, class_combos, hand_class, canon, split

ALL_CLASSES = []
for i in range(12, -1, -1):
    for j in range(12, -1, -1):
        r1, r2 = RANKS[i], RANKS[j]
        if i == j:
            ALL_CLASSES.append(r1 + r2)
        elif i > j:
            ALL_CLASSES.append(r1 + r2 + "s")
        else:
            ALL_CLASSES.append(r2 + r1 + "o")
ALL_CLASSES = list(dict.fromkeys(ALL_CLASSES))


def parse_range(text: str) -> dict:
    out = {}
    for tok in text.strip().split(","):
        tok = tok.strip()
        if not tok:
            continue
        hand, _, w = tok.partition(":")
        out[hand] = float(w) if w else 1.0
    return out


def load_matchup_ranges(key: str):
    """Returns (ip_weights, oop_weights) as {class: weight}."""
    m = preflop.get(key)
    base = os.path.join(config.RANGES_DIR, m["dir"])
    res = []
    for pos in (m["ip"], m["oop"]):
        with open(os.path.join(base, f"{pos}_range.txt")) as f:
            w = parse_range(f.read())
        res.append({h: x for h, x in w.items() if x >= config.PREFLOP_MIN_WEIGHT})
    return res[0], res[1]


def range_string(weights: dict, floor: float = 0.0) -> str:
    parts = [f"{h}:{w:.4f}" for h, w in weights.items() if w > floor]
    return ",".join(parts)


def unblocked(combo: str, dead) -> bool:
    a, b = split(combo)
    return a not in dead and b not in dead


def combo_weights(class_weights: dict, dead=()) -> dict:
    """{combo: weight} for every unblocked combo of every class."""
    dead = set(dead)
    out = {}
    for cls, w in class_weights.items():
        if w <= 0:
            continue
        for c in class_combos(cls):
            if unblocked(c, dead):
                out[c] = w
    return out


def sample_combo(weights: dict, rng=random) -> str:
    combos = list(weights)
    return rng.choices(combos, weights=[weights[c] for c in combos], k=1)[0]


def narrow(reach: dict, board, keep=(), min_weight=None, floor=None) -> dict:
    """Aggregate combo-level reach into class-level weights (all the solver CLI accepts).

    Each class gets the mean reach over its combos not blocked by `board`, normalised so the
    largest class is 1.0. Classes of the hands in `keep` are floored so they stay in the range.
    """
    min_weight = config.RANGE_MIN_WEIGHT if min_weight is None else min_weight
    floor = config.HAND_FLOOR if floor is None else floor
    dead = set(board)
    sums, counts = {}, {}
    for combo, w in reach.items():
        if not unblocked(combo, dead):
            continue
        cls = hand_class(combo)
        sums[cls] = sums.get(cls, 0.0) + w
    for cls in sums:
        counts[cls] = sum(1 for c in class_combos(cls) if unblocked(c, dead))
    out = {cls: sums[cls] / counts[cls] for cls in sums if counts[cls]}
    top = max(out.values(), default=0.0)
    if top <= 0:
        return {}
    out = {cls: w / top for cls, w in out.items()}
    out = {cls: w for cls, w in out.items() if w >= min_weight}
    for combo in keep:
        cls = hand_class(combo)
        out[cls] = max(out.get(cls, 0.0), floor)
    return out
