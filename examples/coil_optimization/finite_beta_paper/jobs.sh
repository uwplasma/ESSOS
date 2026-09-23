#!/bin/zsh
# Usage: jobs.sh JOBFILE PARALLEL. Each line: name<TAB>args... ; every job is killed after 600 s.
# Set PYTHONPATH to the pinned VMEX and SOLVAX first (see versions.json).
cd "$(dirname "$0")"
run_one() {
  local name=$1; shift
  mkdir -p runs/$name
  local t0=$(date +%s)
  eval perl -e "'alarm shift; exec @ARGV'" ${CAP:-600} python -u run.py runs/$name "$@" > runs/$name/run.log 2>&1
  local code=$?
  echo "$name exit=$code seconds=$(( $(date +%s) - t0 ))" >> runs/status.txt
}
typeset -i n=0
while IFS=$'\t' read -r name args; do
  [[ -z $name || $name == \#* ]] && continue
  run_one $name "$args" &
  n+=1
  if (( n % $2 == 0 )); then wait; fi
done < $1
wait
