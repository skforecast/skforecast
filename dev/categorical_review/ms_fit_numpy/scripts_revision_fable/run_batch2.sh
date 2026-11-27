#!/bin/bash
# usage: run_batch2.sh <p2|p3>
cd "<scratchpad>/indep_review"
pd=$1
for tree in base branch; do
  timeout 900 ./run.sh $tree $pd probes.py > logs/probes_${tree}_${pd}.txt 2>&1; echo "probes $tree exit=$?"
  timeout 1200 ./run.sh $tree $pd user_cat_config.py > logs/usercat_${tree}_${pd}.txt 2>&1; echo "usercat $tree exit=$?"
  timeout 600 ./run.sh $tree $pd spy_consistency.py > logs/spy_${tree}_${pd}.txt 2>&1; echo "spy $tree exit=$?"
  timeout 900 ./run.sh $tree $pd others_unchanged.py out/others_${tree}_${pd}.pkl > logs/others_${tree}_${pd}.log 2>&1; echo "others $tree exit=$?"
  for est in lgbm xgb hgb ridge catboost; do
    timeout 900 ./run.sh $tree $pd features_equiv.py $est out/feat_${tree}_${pd}_${est}.pkl > logs/feat_${tree}_${pd}_${est}.log 2>&1; echo "feat $tree $est exit=$?"
    timeout 900 ./run.sh $tree $pd searches.py $est out/search_${tree}_${pd}_${est}.pkl > logs/search_${tree}_${pd}_${est}.log 2>&1; echo "search $tree $est exit=$?"
  done
done
echo ALLDONE
