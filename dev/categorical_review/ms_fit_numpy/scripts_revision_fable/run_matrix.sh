#!/bin/bash
# usage: run_matrix.sh <p2|p3> <tree...>
cd "<scratchpad>/indep_review"
pd=$1; shift
for tree in "$@"; do
  for est in lgbm xgb hgb ridge linreg pipe_ridge pipe_lgbm ttr_ridge ttr_lgbm rf catboost; do
    for mode in wide dict; do
      timeout 900 ./run.sh $tree $pd matrix.py $est out/m_${tree}_${pd}_${est}_${mode}.pkl $mode > logs/m_${tree}_${pd}_${est}_${mode}.log 2>&1
      echo "$tree $pd $est $mode exit=$?"
    done
  done
done
echo ALLDONE
