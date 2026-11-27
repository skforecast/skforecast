#!/bin/bash
# Usage: run_extra.sh  (runs extra.py on base, branch and minimal)
D="<scratchpad>/doubts_b"
S="<scratchpad>"
PY="<conda>/envs/skforecast_27_py13_p3/python.exe"
cd "$D"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
for pair in "base:$S/base028" "branch:<repo>" "minimal:$D/minimal"; do
  tag=${pair%%:*}
  tree=${pair#*:}
  export PYTHONPATH="$tree"
  timeout 900 "$PY" extra.py > "extra_${tag}.txt" 2>&1
  echo "$tag extra exit=$?"
  timeout 600 "$PY" extra.py catboost > "extra_cb_${tag}.txt" 2>&1
  echo "$tag extra catboost exit=$?"
done
