#!/bin/bash
W="<scratchpad>/indep_review"
cd "$W"
# extra probes with pandas 3
for tree in base branch; do
  timeout 900 ./run.sh $tree p3 probes2.py > logs/probes2_${tree}_p3.txt 2>&1; echo "probes2 $tree exit=$?"
  timeout 900 ./run.sh $tree p3 user_cat_config_tuple.py cb > logs/usercat_cbtuple_${tree}_p3.txt 2>&1; echo "cbtuple $tree exit=$?"
done
# save / load: same tree, and saved with base then loaded with the branch
rm -rf saved_branch saved_base
timeout 600 ./run.sh branch p3 save_load.py save saved_branch > logs/save_branch_p3.txt 2>&1; echo "save branch exit=$?"
timeout 600 ./run.sh branch p3 save_load.py load saved_branch > logs/load_branch_branch_p3.txt 2>&1; echo "load branch->branch exit=$?"
timeout 600 ./run.sh base p3 save_load.py save saved_base > logs/save_base_p3.txt 2>&1; echo "save base exit=$?"
timeout 600 ./run.sh branch p3 save_load.py load saved_base > logs/load_base_branch_p3.txt 2>&1; echo "load base->branch exit=$?"
# pandas 2
for tree in base branch; do
  for est in lgbm xgb hgb ridge catboost; do
    for mode in wide dict; do
      timeout 900 ./run.sh $tree p2 matrix.py $est out/m_${tree}_p2_${est}_${mode}.pkl $mode > logs/m_${tree}_p2_${est}_${mode}.log 2>&1
      echo "$tree p2 matrix $est $mode exit=$?"
    done
  done
  timeout 900 ./run.sh $tree p2 probes.py > logs/probes_${tree}_p2.txt 2>&1; echo "probes p2 $tree exit=$?"
  timeout 1200 ./run.sh $tree p2 user_cat_config.py > logs/usercat_${tree}_p2.txt 2>&1; echo "usercat p2 $tree exit=$?"
  timeout 600 ./run.sh $tree p2 spy_consistency.py > logs/spy_${tree}_p2.txt 2>&1; echo "spy p2 $tree exit=$?"
  timeout 900 ./run.sh $tree p2 others_unchanged.py out/others_${tree}_p2.pkl > logs/others_${tree}_p2.log 2>&1; echo "others p2 $tree exit=$?"
  for est in lgbm ridge; do
    timeout 900 ./run.sh $tree p2 features_equiv.py $est out/feat_${tree}_p2_${est}.pkl > logs/feat_${tree}_p2_${est}.log 2>&1; echo "feat p2 $tree $est exit=$?"
  done
  for est in lgbm xgb; do
    timeout 900 ./run.sh $tree p2 searches.py $est out/search_${tree}_p2_${est}.pkl > logs/search_${tree}_p2_${est}.log 2>&1; echo "search p2 $tree $est exit=$?"
  done
done
echo CHAIN4DONE
