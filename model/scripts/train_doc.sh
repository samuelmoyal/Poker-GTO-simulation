#!/bin/bash
# DOC config (15.7M params), longer run: batch 24 avoids an MPS timing cliff at batch>=28 (measured).
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
{
echo "== $(date) train DOC"
$PY -W ignore scripts/train.py --config doc --steps 12000 --batch 24 --lr 3e-4 --warmup 600 --ema 0.995 --eval-every 500 --tag _v1
echo "== $(date) val report"
$PY -W ignore scripts/eval_ckpt.py --split val --ckpt checkpoints/net_turn_doc_v1.pt
echo "== $(date) sensitivity vs solver"
$PY -W ignore scripts/sensitivity.py --n 24 --ckpt checkpoints/net_turn_doc_v1.pt
echo "== $(date) FINAL test split"
$PY -W ignore scripts/eval_ckpt.py --split test --ckpt checkpoints/net_turn_doc_v1.pt
echo "== $(date) done"
} >> data/train_doc.log 2>&1
