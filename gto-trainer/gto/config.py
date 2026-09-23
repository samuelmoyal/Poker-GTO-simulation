"""Central configuration: paths, matchups, solver profile, grading thresholds."""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Preflop range files.  They still live in the TexasSolver release folder, which is only needed for these files.
RANGES_DIR = os.path.join(os.path.dirname(ROOT), "TexasSolver-v0.2.0-MacOs", "ranges", "6max_range")

CACHE_DIR = os.path.join(ROOT, "cache")
FLOP_CACHE_DIR = os.path.join(CACHE_DIR, "flops")
STREET_CACHE_DIR = os.path.join(CACHE_DIR, "streets")
DATA_DIR = os.path.join(ROOT, "data")
WORK_DIR = os.path.join(ROOT, "work")
STATIC_DIR = os.path.join(ROOT, "static")

# The solver rounds bet sizes to integers, so we work in tenths of a big blind.
SCALE = 10

# Heads-up single-raised pots, 100bb deep. `ip`/`oop` are the postflop roles.
# `dir` is relative to RANGES_DIR and holds <POS>_range.txt for both players.
# pot_bb includes dead money (blinds) — that's what the solver needs at the flop.
MATCHUPS = {
    "UTG_vs_BB": dict(label="UTG ouvre, BB call", ip="UTG", oop="BB", dir="UTG/2.5bb/BB/Call", pot_bb=5.5, stack_bb=97.5),
    "MP_vs_BB":  dict(label="MP ouvre, BB call",  ip="MP",  oop="BB", dir="MP/2.5bb/BB/Call",  pot_bb=5.5, stack_bb=97.5),
    "CO_vs_BB":  dict(label="CO ouvre, BB call",  ip="CO",  oop="BB", dir="CO/2.5bb/BB/Call",  pot_bb=5.5, stack_bb=97.5),
    "BTN_vs_BB": dict(label="BTN ouvre, BB call", ip="BTN", oop="BB", dir="BTN/2.5bb/BB/Call", pot_bb=5.5, stack_bb=97.5),
    "UTG_vs_BTN": dict(label="UTG ouvre, BTN call", ip="BTN", oop="UTG", dir="UTG/2.5bb/BTN/Call", pot_bb=6.5, stack_bb=97.5),
    "MP_vs_BTN":  dict(label="MP ouvre, BTN call",  ip="BTN", oop="MP",  dir="MP/2.5bb/BTN/Call",  pot_bb=6.5, stack_bb=97.5),
    "CO_vs_BTN":  dict(label="CO ouvre, BTN call",  ip="BTN", oop="CO",  dir="CO/2.5bb/BTN/Call",  pot_bb=6.5, stack_bb=97.5),
    "SB_vs_BB":  dict(label="SB ouvre, BB call",  ip="BB",  oop="SB", dir="SB/3.0bb/BB/Call",   pot_bb=6.0, stack_bb=97.0),
}

# Bet-size trees, as {solved street: {street: sizes}} (percent of pot; a raise is "to" `raise_`% of the pot after
# the call).  `allin` lists an explicit all-in bet and all-in raise; bets close to the stack are turned into all-ins
# by the solver (`force_allin` in gto/solver.py).  The flop solve is the expensive one, so its turn/river are kept
# minimal (one bet size, no raises, no explicit all-in): that detail barely moves the flop strategy.  The live
# turn/river re-solves use richer trees.
BET_TREE = {
    "flop": {   # used when the root street is the flop (precomputed library)
        "flop":  dict(bet=[50], raise_=[100], donk=[], allin=False),
        "turn":  dict(bet=[66], raise_=[], donk=[66], allin=False),
        "river": dict(bet=[66], raise_=[], donk=[66], allin=False),
    },
    "turn": {   # live turn re-solve
        "turn":  dict(bet=[66], raise_=[150], donk=[66], allin=True),
        "river": dict(bet=[66], raise_=[], donk=[66], allin=True),
    },
    "river": {  # live river re-solve
        "river": dict(bet=[66], raise_=[150], donk=[66], allin=True),
    },
}
THREADS = int(os.environ.get("GTO_THREADS") or os.cpu_count() or 4)

# The solver binary (tools/turn-labels: `cargo build --release --bin street_tree`) and its effort per solved street.
# `target_pct` is the exploitability to reach, in % of the pot at the root of the solved street.  Measured on an
# M2: a turn or river solve takes 0.1-0.6 s to reach 0.5 %.
SOLVER_BIN = os.environ.get("STREET_TREE_BIN") or os.path.join(
    os.path.dirname(ROOT), "tools", "turn-labels", "target", "release", "street_tree")
EFFORT = {
    "flop":  dict(target_pct=0.5, max_iter=1000, max_seconds=300),
    "turn":  dict(target_pct=0.5, max_iter=1000, max_seconds=60),
    "river": dict(target_pct=0.3, max_iter=1000, max_seconds=30),
}

# Hands with a weight below this are dropped: from the preflop range files, and when narrowing ranges between
# streets.  Fewer combos = faster iterations, and the strategy barely moves.
PREFLOP_MIN_WEIGHT = 0.05
RANGE_MIN_WEIGHT = 0.005
# The hero's/villain's actual hand never drops below this weight (keeps the hand in the tree).
HAND_FLOOR = 0.02

# Grading by GTO frequency of the chosen action.
GRADE_OK = 0.30        # >= this: GTO action (pure or a real mix)
GRADE_MIXED = 0.10     # >= this: acceptable, rare mixed action
GRADE_MARGINAL = 0.03  # >= this: marginal; below: error


def profile_id() -> str:
    """Short hash of everything that changes solver output; stale caches are ignored."""
    blob = json.dumps([BET_TREE["flop"], EFFORT["flop"], PREFLOP_MIN_WEIGHT, SCALE, "postflop-solver"], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:6]
