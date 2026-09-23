"""Locations, and access to the trainer's `gto` package (cards, preflop ranges, matchups)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../Poker
TRAINER = os.path.join(ROOT, "gto-trainer")
FLOP_CACHE = os.path.join(TRAINER, "cache", "flops")
LABEL_BIN = os.path.join(ROOT, "tools", "turn-labels", "target", "release", "turn-labels")
BOARDLIB = os.path.join(ROOT, "tools", "boardlib", "target", "release", "libboardlib.dylib")
LOCK_BIN = os.path.join(ROOT, "tools", "turn-labels", "target", "release", "flop_lock")
FLOP_BIN = os.path.join(ROOT, "tools", "turn-labels", "target", "release", "flop_lines")
DATA_DIR = os.path.join(ROOT, "model", "data")
TURN_LABELS_DIR = os.path.join(DATA_DIR, "turn_labels")
NATIVE_FLOPS = os.path.join(DATA_DIR, "native_flops")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")

if TRAINER not in sys.path:
    sys.path.insert(0, TRAINER)
