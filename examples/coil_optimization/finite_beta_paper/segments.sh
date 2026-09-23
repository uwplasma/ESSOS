#!/bin/zsh
# Usage: segments.sh NAME ARGS... : repeat 10-minute optimization segments until finished.
# Set PYTHONPATH to the pinned VMEX and SOLVAX first (see versions.json).
cd "$(dirname "$0")"
name=$1; shift
mkdir -p runs/$name
for i in {1..30}; do
  t0=$(date +%s)
  perl -e 'alarm shift; exec @ARGV' 600 python -u run.py runs/$name "RUN_VMEX = False" "EVALUATIONS_PER_RUN = 40" "$@" > runs/$name/segment_$i.log 2>&1
  code=$?
  echo "$name segment=$i exit=$code seconds=$(( $(date +%s) - t0 ))" >> runs/status.txt
  grep -q "^Reusing" runs/$name/segment_$i.log && break
  (( code != 0 )) && ! grep -q "Continuing" runs/$name/segment_$i.log && [[ $i -gt 1 ]] && break
done
