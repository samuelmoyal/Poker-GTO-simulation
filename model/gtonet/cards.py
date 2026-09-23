"""Card / combo indexing (52 cards, 1326 combos) and the 24 suit permutations.

Card id = 4 * rank + suit with ranks "23456789TJQKA" and suits "cdhs" (same order as `gto.cards.DECK`).
Combo index = position of the pair (i < j) in lexicographic order.  Every per-hand vector in the dataset
uses this order; a suit permutation is applied to a vector with `permute_combos`.
"""
import hashlib
import itertools

import numpy as np

from . import paths  # noqa: F401  (puts gto-trainer on sys.path)
from gto.cards import DECK, canonical_flop  # noqa: E402

NUM_CARDS, NUM_COMBOS = 52, 1326
CARD_ID = {c: i for i, c in enumerate(DECK)}
COMBOS = [(i, j) for i in range(NUM_CARDS) for j in range(i + 1, NUM_CARDS)]
COMBO_ID = {p: k for k, p in enumerate(COMBOS)}
COMBO_CARDS = np.array(COMBOS, dtype=np.int64)  # (1326, 2)


def parse_hand(hand: str):
    a, b = CARD_ID[hand[:2]], CARD_ID[hand[2:]]
    return (a, b) if a < b else (b, a)


def combo_index(hand: str) -> int:
    return COMBO_ID[parse_hand(hand)]


def combo_str(k: int) -> str:
    i, j = COMBOS[k]
    return DECK[i] + DECK[j]


def blocked_mask(dead_cards) -> np.ndarray:
    """(1326,) bool, True where the combo contains a dead card."""
    dead = np.zeros(NUM_CARDS, dtype=bool)
    for c in dead_cards:
        dead[CARD_ID[c] if isinstance(c, str) else c] = True
    return dead[COMBO_CARDS].any(axis=1)


SUIT_PERMS = list(itertools.permutations(range(4)))  # perm[s] = new suit of suit s


def card_perm(perm) -> np.ndarray:
    """new_card_id[old_card_id]."""
    return np.array([4 * (c // 4) + perm[c % 4] for c in range(NUM_CARDS)], dtype=np.int64)


def combo_perm(perm) -> np.ndarray:
    """new_combo_index[old_combo_index]."""
    cp = card_perm(perm)
    return np.array([COMBO_ID[tuple(sorted((cp[i], cp[j])))] for i, j in COMBOS], dtype=np.int64)


def permute_combos(vec: np.ndarray, cperm: np.ndarray) -> np.ndarray:
    """Relabel a per-combo vector (last axis 1326) under a suit permutation."""
    out = np.empty_like(vec)
    out[..., cperm] = vec
    return out


def canonical_flop_key(board) -> str:
    """Suit-isomorphism class of a flop, e.g. 'KsQhJd' (any 3 of the first cards of `board`)."""
    return "".join(canonical_flop(list(board)[:3])[0])


def split_of(board) -> str:
    """train / val / test by hash of the canonical flop, so isomorphic boards and every matchup, line,
    turn card and perturbation of a flop always land on the same side."""
    h = int(hashlib.sha1(("flop:" + canonical_flop_key(board)).encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else "val" if h < 90 else "test"
