#!/bin/bash
# Floor -> train -> val report -> learning curve -> sensitivity -> test, on the featurised dataset.  Log: data/analysis.log
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
{
echo "== $(date) equity floor";    $PY -W ignore scripts/eval_baseline.py val
echo "== $(date) train";           $PY -W ignore scripts/train.py --config small --steps 6000 --batch 32 --lr 5e-4 --eval-every 500 --ema 0.995
echo "== $(date) val report";      $PY -W ignore scripts/eval_ckpt.py --split val
echo "== $(date) learning curve";  $PY -W ignore scripts/learning_curve.py --steps 3000
echo "== $(date) sensitivity vs solver"; $PY -W ignore scripts/sensitivity.py --n 24
echo "== $(date) FINAL test split (report only, never tune on it)"; $PY -W ignore scripts/eval_ckpt.py --split test
echo "== $(date) done"
} >> data/analysis.log 2>&1
