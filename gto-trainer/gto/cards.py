"""Cards, hand classes, suit canonicalisation and a 7-card evaluator."""
import random
from collections import Counter

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RVAL = {r: i for i, r in enumerate(RANKS)}
DECK = [r + s for r in RANKS for s in SUITS]


def _key(card):
    return (RVAL[card[0]], SUITS.index(card[1]))


def canon(combo: str) -> str:
    """Two-card combo in a single canonical order: higher rank first, then higher suit."""
    a, b = combo[:2], combo[2:]
    return a + b if _key(a) >= _key(b) else b + a


def hand_class(combo: str) -> str:
    c = canon(combo)
    if c[0] == c[2]:
        return c[0] + c[2]
    return c[0] + c[2] + ("s" if c[1] == c[3] else "o")


def class_combos(cls: str):
    """All canonical combos of a hand class such as 'AKs', 'QQ', 'T9o'."""
    r1, r2 = cls[0], cls[1]
    out = []
    if r1 == r2:
        for i, s1 in enumerate(SUITS):
            for s2 in SUITS[i + 1:]:
                out.append(canon(r1 + s1 + r2 + s2))
    elif cls[2] == "s":
        out = [canon(r1 + s + r2 + s) for s in SUITS]
    else:
        out = [canon(r1 + s1 + r2 + s2) for s1 in SUITS for s2 in SUITS if s1 != s2]
    return out


def split(combo: str):
    return combo[:2], combo[2:]


def canonical_flop(board):
    """Relabel suits by first appearance (rank-descending). Returns (canonical board, real->canon suit map)."""
    ordered = sorted(board, key=_key, reverse=True)
    order = "shdc"
    mapping = {}
    for card in ordered:
        if card[1] not in mapping:
            mapping[card[1]] = order[len(mapping)]
    return [c[0] + mapping[c[1]] for c in ordered], mapping


def map_combo(combo: str, suit_map: dict) -> str:
    a, b = split(combo)
    return canon(a[0] + suit_map.get(a[1], a[1]) + b[0] + suit_map.get(b[1], b[1]))


def random_suit_map():
    perm = list(SUITS)
    random.shuffle(perm)
    return dict(zip(SUITS, perm))


def _straight_high(values):
    s = set(values)
    if 12 in s:
        s.add(-1)
    for high in range(12, 2, -1):
        if all((high - i) in s for i in range(5)):
            return high
    return None


def evaluate(cards):
    """Best 5-card hand out of up to 7 cards, as a comparable tuple (higher wins)."""
    vals = sorted((RVAL[c[0]] for c in cards), reverse=True)
    by_suit = {}
    for c in cards:
        by_suit.setdefault(c[1], []).append(RVAL[c[0]])
    flush = next((sorted(v, reverse=True) for v in by_suit.values() if len(v) >= 5), None)
    if flush:
        sf = _straight_high(flush)
        if sf is not None:
            return (8, sf)
    counts = Counter(vals)
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    quads = [v for v, n in groups if n == 4]
    trips = [v for v, n in groups if n == 3]
    pairs = [v for v, n in groups if n == 2]
    if quads:
        kicker = max(v for v in vals if v != quads[0])
        return (7, quads[0], kicker)
    if trips and (len(trips) > 1 or pairs):
        pair = max(trips[1:] + pairs)
        return (6, trips[0], pair)
    if flush:
        return (5, *flush[:5])
    st = _straight_high(vals)
    if st is not None:
        return (4, st)
    if trips:
        kick = [v for v in vals if v != trips[0]][:2]
        return (3, trips[0], *kick)
    if len(pairs) >= 2:
        top = pairs[:2]
        kicker = max(v for v in vals if v not in top)
        return (2, top[0], top[1], kicker)
    if pairs:
        kick = [v for v in vals if v != pairs[0]][:3]
        return (1, pairs[0], *kick)
    return (0, *vals[:5])


def describe(score):
    names = ["Carte haute", "Paire", "Double paire", "Brelan", "Quinte", "Couleur", "Full", "Carré", "Quinte flush"]
    return names[score[0]]
