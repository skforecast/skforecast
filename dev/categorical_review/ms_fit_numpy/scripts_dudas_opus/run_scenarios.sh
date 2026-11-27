#!/bin/bash
# Usage: run_scenarios.sh <tag> <tree_path>
D="<scratchpad>/doubts_b"
PY="<conda>/envs/skforecast_27_py13_p3/python.exe"
tag=$1
tree=$2
cd "$D"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$tree"
for fam in lgbm xgb hgb ridge pipe_lgbm ttr_lgbm catboost; do
  timeout 900 "$PY" scenarios.py $fam "res_${tag}_${fam}.json" > "log_${tag}_${fam}.txt" 2>&1
  echo "$tag $fam exit=$?"
done
