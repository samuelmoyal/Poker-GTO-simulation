"""Vectorised hand strength for 5-7 cards (numpy), replacing the slow per-hand `gto.cards.evaluate`.

Cards are ids 0..51 (`cards.DECK` order: 4 * rank + suit).  `strength` returns an int64 whose order is the
poker order (equal score = tie).  Score = category * 14**5 + five base-14 digits (rank + 1, 0 if unused):
  8 straight flush, 7 quads, 6 full house, 5 flush, 4 straight, 3 trips, 2 two pair, 1 pair, 0 high card.
"""
import numpy as np

RANKS = np.arange(13)
CAT_BASE = 14 ** 5


def _straight_high(p: np.ndarray) -> np.ndarray:
    """p: (N, 13) rank presence -> high rank of the best straight (wheel = 3), or -1."""
    out = np.full(len(p), -1, dtype=np.int64)
    out[p[:, 12] & p[:, 0] & p[:, 1] & p[:, 2] & p[:, 3]] = 3
    for high in range(4, 13):
        out[p[:, high - 4:high + 1].all(1)] = high
    return out


def _topk(mask: np.ndarray, k: int) -> np.ndarray:
    """(N, 13) bool -> (N, k) highest ranks present, descending, -1 when fewer."""
    v = np.where(mask, RANKS, -1)
    return -np.sort(-v, axis=1)[:, :k]


def _pack(cat: int, cols) -> np.ndarray:
    score = np.full(len(cols[0]), cat, dtype=np.int64)
    for i in range(5):
        score = score * 14 + ((cols[i] + 1) if i < len(cols) else 0)
    return score


def strength(cards: np.ndarray) -> np.ndarray:
    cards = np.asarray(cards)
    n, k = cards.shape
    r, s = cards // 4, cards % 4
    cnt = (r[:, :, None] == RANKS).sum(1)                       # (N, 13)
    present = cnt > 0
    suit_cnt = (s[:, :, None] == np.arange(4)).sum(1)            # (N, 4)
    suit_present = np.zeros((n, 4, 13), dtype=bool)
    for i in range(k):
        suit_present[np.arange(n), s[:, i], r[:, i]] = True
    has_flush = suit_cnt.max(1) >= 5
    flush = suit_present[np.arange(n), suit_cnt.argmax(1)] & has_flush[:, None]
    st, st_fl = _straight_high(present), _straight_high(flush)

    quad = np.where(cnt == 4, RANKS, -1).max(1)
    t1 = np.where(cnt >= 3, RANKS, -1).max(1)
    pair_ex_t1 = np.where((cnt >= 2) & (RANKS != t1[:, None]), RANKS, -1).max(1)  # best pair besides the top trips
    p1 = np.where(cnt >= 2, RANKS, -1).max(1)
    p2 = np.where((cnt >= 2) & (RANKS != p1[:, None]), RANKS, -1).max(1)

    def kick(excluded, m):  # top-m ranks not in `excluded` (list of (N,) arrays)
        mask = present.copy()
        for e in excluded:
            mask &= RANKS != e[:, None]
        return _topk(mask, m)

    fl5 = _topk(flush, 5)
    tr = kick([t1], 2)
    tp = kick([p1, p2], 1)
    pr = kick([p1], 3)
    hi = _topk(present, 5)
    scores = [
        (st_fl >= 0, _pack(8, [st_fl])),
        (quad >= 0, _pack(7, [quad, kick([quad], 1)[:, 0]])),
        ((t1 >= 0) & (pair_ex_t1 >= 0), _pack(6, [t1, pair_ex_t1])),
        (has_flush, _pack(5, [fl5[:, i] for i in range(5)])),
        (st >= 0, _pack(4, [st])),
        (t1 >= 0, _pack(3, [t1, tr[:, 0], tr[:, 1]])),
        (p2 >= 0, _pack(2, [p1, p2, tp[:, 0]])),
        (p1 >= 0, _pack(1, [p1, pr[:, 0], pr[:, 1], pr[:, 2]])),
    ]
    return np.select([c for c, _ in scores], [v for _, v in scores], default=_pack(0, [hi[:, i] for i in range(5)]))


def category(score: np.ndarray) -> np.ndarray:
    return score // CAT_BASE
