"""Flop library: solved once per (matchup, suit-canonical flop), reused with any suit permutation."""
import os
import random

from . import config, preflop, ranges, solver
from .cards import DECK, canonical_flop


def flop_path(matchup: str, canon_board) -> str:
    name = f"{matchup}__{''.join(canon_board)}__{config.profile_id()}.json.gz"
    return os.path.join(config.FLOP_CACHE_DIR, name)


def solve_flop(matchup: str, canon_board, on_progress=None):
    m = preflop.get(matchup)
    ip_w, oop_w = ranges.load_matchup_ranges(matchup)
    return solver.solve(
        board=list(canon_board),
        pot=m["pot_bb"] * config.SCALE,
        stack=m["stack_bb"] * config.SCALE,
        ip_range=ranges.range_string(ip_w),
        oop_range=ranges.range_string(oop_w),
        street="flop",
        cache_path=flop_path(matchup, canon_board),
        on_progress=on_progress,
    )


def cached_flops(matchup=None):
    """[(matchup, canonical board list)] currently in the library."""
    out = []
    if not os.path.isdir(config.FLOP_CACHE_DIR):
        return out
    for name in sorted(os.listdir(config.FLOP_CACHE_DIR)):
        if not name.endswith(".json.gz"):  # skips .meta.json sidecars
            continue
        mk, board, pid = name[: -len(".json.gz")].split("__")
        if pid != config.profile_id() or (matchup and mk != matchup):
            continue
        out.append((mk, [board[i:i + 2] for i in range(0, 6, 2)]))
    return out


def random_flop(rng=random):
    return rng.sample(DECK, 3)


def random_canonical_flop(rng=random):
    return canonical_flop(random_flop(rng))[0]
