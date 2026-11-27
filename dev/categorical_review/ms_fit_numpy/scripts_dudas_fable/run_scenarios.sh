#!/bin/bash
# Usage: run_scenarios.sh <label> <tree path>
W="<scratchpad>/doubts_a"
PY=<conda>/envs/skforecast_27_py13_p3/python.exe
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$2"
cd "$W"
for kind in wide dict; do
  for fam in lgbm xgb hgb ridge pipe_lgbm ttr_lgbm ttr_xgb catboost; do
    timeout 900 $PY scenarios.py $fam out/$1_${fam}_${kind}.json $kind > out/$1_${fam}_${kind}.log 2>&1
    echo "$1 $fam $kind exit=$?"
  done
done
