#!/bin/bash
# Round 2: ~100-120 new flops + a bounded batch of states (new range-kind mix: on-policy/noise/structured)
# spread over ALL cached flops (old + new), then retrain and re-evaluate. Log: data/pipeline_v2.log
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
{
echo "== $(date) gen_flops (new flops, seed 1)"
$PY -u scripts/gen_flops.py --total 120 --seed 1
echo "== $(date) gen_labels (old+new flops, new range-kind mix, capped)"
$PY -u scripts/gen_labels.py --n-per-flop 16 --max-states 900 --seed 1
echo "== $(date) backfill + featurise (new shards only, single process: parallel MPS workers hung last time)"
$PY -W ignore scripts/backfill_v2.py
$PY -W ignore scripts/featurize.py
echo "== $(date) train DOC v2"
$PY -W ignore scripts/train.py --config doc --steps 12000 --batch 24 --lr 3e-4 --warmup 600 --ema 0.995 --eval-every 500 --tag _v2
echo "== $(date) val report"
$PY -W ignore scripts/eval_ckpt.py --split val --ckpt checkpoints/net_turn_doc_v2.pt
echo "== $(date) sensitivity vs solver"
$PY -W ignore scripts/sensitivity.py --n 24 --ckpt checkpoints/net_turn_doc_v2.pt
echo "== $(date) FINAL test split"
$PY -W ignore scripts/eval_ckpt.py --split test --ckpt checkpoints/net_turn_doc_v2.pt
echo "== $(date) done"
} >> data/pipeline_v2.log 2>&1
