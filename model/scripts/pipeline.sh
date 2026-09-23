#!/bin/bash
# Wait for the data generation to finish, then: backfill -> featurise -> floor -> train -> learning curve -> sensitivity -> test.
# Log: data/pipeline.log.  Usage: caffeinate -i bash scripts/pipeline.sh
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
{
echo "== $(date) waiting for gen_flops / gen_labels to finish"
while true; do
  if pgrep -f "scripts/gen_(flops|labels).py" > /dev/null; then sleep 60; continue; fi
  sleep 30
  pgrep -f "scripts/gen_(flops|labels).py" > /dev/null || break   # not just the gap between the two stages
done
echo "== $(date) backfill";        $PY -W ignore scripts/backfill_v2.py
echo "== $(date) featurise (6 workers)"; $PY -W ignore scripts/featurize.py 6 || $PY -W ignore scripts/featurize.py 1
bash scripts/analysis.sh
} >> data/pipeline.log 2>&1
