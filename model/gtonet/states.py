"""Turn-root states: on-policy ranges harvested from flop solves, plus two kinds of off-policy ranges.

A state is everything the solver needs (and the net will see): 4-card board, pot, effective stack, and both
players' per-combo ranges.  Sampling is deterministic given (seed, flop file), so re-running a generation
reproduces the same ids and can resume.

Each side's range is drawn independently (`RANGE_KIND`) from:
  on-policy   the real reach at this line, from the flop solve
  noise       on-policy + multiplicative log-normal noise + a blend with uniform (`perturb`)
  structured  a linear or polarised range by hand-strength rank on this board (`structured`), unrelated to
              the line's actual reach

`noise` covers imprecise/nodelocked ranges close to a real one; `structured` covers what a value net actually
gets tested on (architecture.md §11's range-sensitivity test) and on-policy solves rarely produce on their
own: equilibrium ranges are continuous mixed frequencies, not the sharp linear/polarised shapes a non-GTO or
exploitative opponent can present. Independent per side so training sees "one side normal, other side weird",
the shape of the sensitivity test itself.
"""
import glob
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass

import numpy as np

from . import cards, flopdump, nativeflops, paths, poker
from gto import preflop  # noqa: E402
from gto.cards import DECK  # noqa: E402

MIN_WEIGHT = 0.002   # combos below this (relative to the range's max) are dropped
RANGE_KIND = {"onpolicy": 0.70, "noise": 0.15, "structured": 0.15}  # per side, independently
LINE_MASS_POWER = 0.5  # lines are sampled with probability ~ mass**0.5 (flatter than on-policy frequency)


@dataclass
class TurnState:
    id: str
    matchup: str
    kind: str
    board: list      # 4 cards, e.g. ["Td", "9d", "6h", "Qc"]
    pot: int         # chips
    stack: int       # chips
    line: str
    perturbed: bool
    split: str
    oop: dict        # {hand: weight in (0, 1]}
    ip: dict
    source: int = 0  # 1 = TexasSolver flop library, 2 = native flop solve
    range_kind: tuple = ("onpolicy", "onpolicy")  # (oop, ip); debug/reporting only, not stored in the schema

    def request(self, target_pct: float) -> dict:
        return dict(id=self.id, flop="".join(self.board[:3]), turn=self.board[3], pot=self.pot, stack=self.stack,
                    oop=self.oop, ip=self.ip, target_pct=target_pct)


def _normalise(weights: dict, dead) -> dict:
    top = max((w for h, w in weights.items() if not (set(cards.parse_hand(h)) & dead)), default=0.0)
    if top <= 0:
        return {}
    out = {}
    for h, w in weights.items():
        if set(cards.parse_hand(h)) & dead:
            continue
        if w / top >= MIN_WEIGHT:
            out[h] = min(1.0, w / top)
    return out


def perturb(weights: dict, board, rng: random.Random) -> dict:
    """Multiplicative log-normal noise, then a blend with the uniform range over unblocked combos."""
    dead = {cards.CARD_ID[c] for c in board}
    sigma, lam = rng.uniform(0.15, 0.7), rng.uniform(0.0, 0.15)
    top = max(weights.values())
    noisy = {h: w / top * math.exp(sigma * rng.gauss(0, 1)) for h, w in weights.items()}
    top = max(noisy.values())
    out = {}
    for k in range(cards.NUM_COMBOS):
        if set(cards.COMBOS[k]) & dead:
            continue
        h = cards.combo_str(k)
        out[h] = (1 - lam) * noisy.get(h, 0.0) / top + lam
    return out


def _hand_order(board) -> np.ndarray:
    """(1326,) made-hand strength (hole + 4-card board, `poker.strength`'s order), -1 for blocked combos.

    A cheap, deterministic proxy for on-board hand rank (no equity/MPS needed): good enough to build a
    linear/polarised *shape*, which only needs a strength ordering, not exact equity.
    """
    dead = {cards.CARD_ID[c] for c in board}
    ok = ~np.isin(cards.COMBO_CARDS, list(dead)).any(1)
    board_ids = np.array([cards.CARD_ID[c] for c in board])
    hands = np.concatenate([cards.COMBO_CARDS, np.tile(board_ids, (cards.NUM_COMBOS, 1))], axis=1)
    return np.where(ok, poker.strength(hands), -1)


def structured(board, rng: random.Random) -> dict:
    """A linear (top ~50%) or polarised (top ~15% + bottom ~15%) range by hand-strength rank on this board.

    Ignores the line's actual reach entirely: this is the "very different range" of the sensitivity test
    (architecture.md §11), generated at training time instead of only at evaluation time.
    """
    order = _hand_order(board)
    live = np.flatnonzero(order >= 0)
    ranked = live[np.argsort(-order[live])]
    n = len(ranked)
    w = np.zeros(cards.NUM_COMBOS)
    if rng.random() < 0.5:
        w[ranked[: max(1, int(rng.uniform(0.35, 0.65) * n))]] = 1.0
    else:
        top, bot = rng.uniform(0.08, 0.25), rng.uniform(0.05, 0.20)
        w[ranked[: max(1, int(top * n))]] = 1.0
        w[ranked[n - max(1, int(bot * n)):]] = 1.0
    # sparse smoothing so the cutoff isn't a razor-sharp mask: a FEW extra combos outside it, not a floor on
    # every unblocked combo (that made the "structured" range ~1128/1326 wide, as heavy to solve as a full
    # range and 5-10x slower than the intended sparse shape -- caught from the generation log, not a test)
    extra_p = rng.uniform(0.0, 0.06)
    for k in ranked:
        if w[k] == 0.0 and rng.random() < extra_p:
            w[k] = rng.uniform(0.05, 0.3)
    return {cards.combo_str(k): float(w[k]) for k in live if w[k] > 1e-6}


def _side_range(reach: dict, board, rng: random.Random) -> tuple:
    kind = rng.choices(list(RANGE_KIND), list(RANGE_KIND.values()))[0]
    r = reach if kind == "onpolicy" else perturb(reach, board, rng) if kind == "noise" else structured(board, rng)
    return r, kind


def _state_id(matchup, board, line, perturbed, nonce) -> str:
    return hashlib.sha1(f"{matchup}|{''.join(board)}|{line}|{int(perturbed)}|{nonce}".encode()).hexdigest()[:16]


def flop_states(path: str, n: int, seed=0):
    """Up to `n` distinct (line, turn card) states from one flop solve."""
    matchup, flop, _ = flopdump.parse_name(path)
    rng = random.Random(f"{seed}:{os.path.basename(path)}")
    native = path.endswith(f"__{nativeflops.PID}.json.gz")
    if native:
        lines = nativeflops.load_lines(path)
    else:
        lines = flopdump.replay(flopdump.load(path), matchup, flop)
    lines = [l for l in lines if l.stack > 0 and l.pot > 0]
    if not lines:
        return []
    weights = [max(l.mass, 1e-9) ** LINE_MASS_POWER for l in lines]
    kind = preflop.get(matchup)["kind"]
    seen, out = set(), []
    for _ in range(n * 4):
        if len(out) >= n:
            break
        fl = rng.choices(lines, weights)[0]
        turn = rng.choice([c for c in DECK if c not in flop])
        if (fl.line, turn) in seen:
            continue
        seen.add((fl.line, turn))
        board = flop + [turn]
        dead = {cards.CARD_ID[c] for c in board}
        r_oop, k_oop = _side_range(fl.reach[flopdump.OOP], board, rng)
        r_ip, k_ip = _side_range(fl.reach[flopdump.IP], board, rng)
        perturbed = k_oop != "onpolicy" or k_ip != "onpolicy"
        oop, ip = _normalise(r_oop, dead), _normalise(r_ip, dead)
        if len(oop) < 2 or len(ip) < 2:
            continue
        out.append(TurnState(_state_id(matchup, board, fl.line, perturbed, rng.getrandbits(32)), matchup, kind,
                             board, fl.pot, fl.stack, fl.line, perturbed, cards.split_of(board), oop, ip,
                             2 if native else 1, (k_oop, k_ip)))
    return out


def flop_files():
    """One file per (matchup, canonical flop): the best converged one when several profiles exist."""
    best = {}
    cached = glob.glob(os.path.join(paths.FLOP_CACHE, "*.json.gz")) + glob.glob(os.path.join(paths.NATIVE_FLOPS, "*.json.gz"))
    for p in cached:
        try:
            matchup, flop, _ = flopdump.parse_name(p)
            preflop.get(matchup)
            with open(p.replace(".json.gz", ".meta.json")) as f:
                expl = json.load(f).get("exploitability") or 99.0
        except (KeyError, ValueError, OSError):
            continue
        key = (matchup, "".join(flop))
        if p.endswith(f"__{nativeflops.PID}.json.gz"):
            expl = -1.0  # native solves are preferred over TexasSolver ones for the same (matchup, flop)
        if key not in best or expl < best[key][0]:
            best[key] = (expl, p)
    return sorted(p for _, p in best.values())
