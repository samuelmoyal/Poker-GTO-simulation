#!/bin/bash
# Same data (v2 dataset), same protocol, three head widths: does a smaller net lose accuracy? Log: data/compare_sizes.log
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
{
for cfg in small s2 tiny; do
  echo "== $(date) train $cfg"
  $PY -W ignore scripts/train.py --config $cfg --steps 6000 --batch 32 --lr 5e-4 --ema 0.995 --eval-every 1000 --tag _v2 | grep -E "params|^step|FINAL"
  echo "== $(date) test $cfg"
  $PY -W ignore scripts/eval_ckpt.py --split test --ckpt checkpoints/net_turn_${cfg}_v2.pt | head -3
  echo "== $(date) sensitivity $cfg"
  $PY -W ignore scripts/sensitivity.py --n 24 --ckpt checkpoints/net_turn_${cfg}_v2.pt
done
echo "== $(date) done"
} >> data/compare_sizes.log 2>&1
