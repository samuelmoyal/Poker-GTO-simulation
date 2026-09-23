# Poker: GTO trainer + real-time flop solving with a value network

Two things live in this repository:

1. **`gto-trainer/`** — a local web app that plays heads-up NLHE hands against a solver-driven bot and grades each of your
   decisions against solver strategies.
2. **`model/` + `tools/`** — a research track to solve **flops in real time on an Apple Silicon Mac**. Exact solves of
   turn roots (from `postflop-solver`) are distilled into a value network, `net_turn`, which stands in for everything that
   comes after the flop inside a truncated CFR. Design and rationale: [`architecture.md`](architecture.md); working rules:
   [`CLAUDE.md`](CLAUDE.md).

## Where things stand

| | Measured |
|---|---|
| Label generation | exact turn-root solve ≈ 1.5–2.6 s (native solver; TexasSolver ≈ 15–30 s); flop ≈ 40–60 s |
| `net_turn` accuracy (range-weighted MAE, % of pot, held-out flops) | equity baseline 18.6 → DOC 4.6 / SMALL 5.0 / S2 5.6 / TINY 6.1 |
| Real EV loss of the resolved flop strategy (one held-out spot, exact continuation) | SMALL 0.64 %, DOC 0.74 %, TINY 0.92 % of the pot; equity-only leaves 3.11 %; uniform 20.9 % |
| One flop decision (TINY, 100 CFR iterations, Apple M2) | ≈ 4.5 s, against a 0.4 s target |

The real-EV-loss figure is a single spot and a single run (solver noise ≈ ±0.15 % of the pot); see `architecture.md` §11–12
for the caveats and the other measurements. The resolver and the network are **not** wired into the trainer app yet.

## Layout

| Path | What |
|---|---|
| `gto-trainer/` | Web trainer (stdlib Python + vanilla JS). Uses TexasSolver for now. See its own README (French). |
| `model/gtonet/` | Python package: cards/evaluator/equity, label pipeline, `net_turn`, vector CFR, real-time resolver |
| `model/scripts/` | Data generation, featurisation, training, evaluation, experiment runners |
| `model/tests/` | `unittest` suite (35 tests) |
| `model/checkpoints/` | The small trained nets (TINY, S2, SMALL); the 63 MB DOC nets are not versioned |
| `tools/postflop-solver/` | Vendored Rust solver (AGPL-3.0, see `VENDORED.md`) |
| `tools/turn-labels/` | Rust JSONL front-ends: `turn-labels`, `flop_lines`, `flop_lock` |
| `tools/boardlib/` | Rust library (C ABI, used through `ctypes`): 7-card strength, win tables, equity |
| `bench/` | Micro-benchmarks and validation experiments |

## Requirements

Apple Silicon Mac (training and inference use PyTorch's `mps` backend), Python 3.13, Rust (`brew install rust`).

## Setup

```bash
# 1. Rust tools (postflop-solver is built as a dependency)
(cd tools/turn-labels && cargo build --release)
(cd tools/boardlib && cargo build --release)

# 2. Python environment for the model track (the trainer itself needs nothing beyond the standard library)
python3 -m venv model/.venv && model/.venv/bin/pip install numpy pyarrow torch

# 3. TexasSolver (needed by the trainer, and for the preflop range files the model track reads).
#    Download the macOS build of v0.2.0 from https://github.com/bupticybee/TexasSolver and unzip it as
#    ./TexasSolver-v0.2.0-MacOs  (git-ignored; it is a third-party release with its own license).
```

## Running

```bash
cd gto-trainer && python3 server.py                     # trainer on http://localhost:8765

cd model
.venv/bin/python -m unittest discover -s tests          # tests
.venv/bin/python scripts/gen_flops.py --total 300       # native flop solves -> data/native_flops   (hours)
.venv/bin/python scripts/gen_labels.py --n-per-flop 24  # exact turn-root labels -> data/turn_labels (hours)
.venv/bin/python scripts/backfill_v2.py && .venv/bin/python scripts/featurize.py
.venv/bin/python scripts/eval_baseline.py val           # the equity floor to beat
.venv/bin/python scripts/train.py --config small --steps 6000
# evaluation, learning curve, sensitivity, resolver benchmarks: see model/README.md and bench/
```

Generated data (`model/data`, caches, logs) is git-ignored and regenerable; the pipeline is resumable.

## Planned: remove TexasSolver

TexasSolver is a temporary dependency. Two things tie us to it, and both need handling:

1. **The solver itself**, used by the trainer for live turn/river solves and the flop library. `postflop-solver` already does
   everything the trainer needs, faster and with per-combo ranges and per-action EVs. Missing: an exporter that dumps a solved
   street in the tree format `gto/engine.py` reads (a tree walk like `flop_lock`'s), then a swap in `gto/solver.py`, and a
   parity check of strategies on a few spots (all-in rules differ between the two solvers).
2. **Its data**: the preflop range files under `ranges/6max_range` (1483 files) that define every spot. They must be kept or
   replaced before the folder can go; their provenance and licence need checking first.

## License

Not chosen yet. Note that `tools/postflop-solver` is AGPL-3.0-or-later and `tools/turn-labels` links it, so that binary is
subject to the same terms; decide on the repository license before publishing.
