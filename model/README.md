# model/ — value net data pipeline (turn roots)

Labels for `net_turn`: exact turn-root solves (through the river) by `../tools/turn-labels`, a thin JSONL
wrapper around `../tools/postflop-solver` (Rust). Ranges at the turn root come from the flop solves cached by
the trainer (`../gto-trainer/cache/flops`): 70% on-policy per-combo ranges, 30% perturbed.

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pyarrow torch          # once
(cd ../tools/turn-labels && cargo build --release)                          # once (needs Rust: brew install rust)
(cd ../tools/boardlib && cargo build --release)                             # once: fast per-board tables (used by resolver.py)
.venv/bin/python scripts/gen_labels.py --n-per-flop 24                      # resumable, writes data/turn_labels/*.parquet
.venv/bin/python scripts/gen_flops.py --total 300                          # native flop solves -> data/native_flops
.venv/bin/python scripts/backfill_v2.py && .venv/bin/python scripts/featurize.py   # labels -> featurised dataset
.venv/bin/python scripts/eval_baseline.py                                   # the equity floor (CLAUDE.md: before training)
.venv/bin/python scripts/train.py --overfit 16 --steps 800                  # plumbing check, then --config small --steps 5000
.venv/bin/python -m unittest discover -s tests
.venv/bin/python ../bench/bench_labels.py 20
```

Schema: see `gtonet/labels.py`. Splits (train 80 / val 10 / test 10) are by hash of the *canonical flop*, so every
matchup, line, turn card, perturbation and suit-isomorphic copy of a flop stays on one side. Val/test labels are
solved to 0.1% of pot (train: 0.5%) so held-out error is not dominated by solver noise.

`postflop-solver` needs `--no-default-features --features rayon` and a lint allowance (see its
`.cargo/config.toml`); its "normalized weights" sum to the number of valid card pairs, not 1 — renormalise.

## Code map
`cards` (1326 combos, 24 suit perms) - `poker` (vectorised 7-card strength) - `equity` (per-board hand-vs-hand tables,
conditional equity, card-removal corrected marginals) - `flopdump`/`nativeflops`/`states` (turn-root states) -
`labels` (solver wrapper, Parquet schema v2) - `dataset` (features) - `batching` (canonical suits, augmentation) -
`model` (`NetTurn`, zero-sum layer, loss) - `evaluate` (range-weighted MAE, slices) - `sensitivity` (range test).
Units: stored `v_*` are EV/pot; the net predicts `(EV/pot - 1/2)/(1+SPR)`; MAE is reported in % of pot (EV/pot units).

## Runtime pieces (real-time flop resolving)
`cfr` (vector DCFR, numpy) - `resolver` (truncated flop CFR, `NetLeaves`) - `oracle` (exact leaves from the solver, no net) -
`fastboard` (ctypes front-end of `tools/boardlib`, Rust: 7-card strength, per-board win tables, equity by sorting).
`NetLeaves` evaluates only the initial range's hands, renames suits instead of permuting vectors, skips the policy head
(all exact, see tests). Benchmarks: `bench/bench_resolve.py`, `bench/validate_resolver.py`, `bench/bench_boardlib.py`.
