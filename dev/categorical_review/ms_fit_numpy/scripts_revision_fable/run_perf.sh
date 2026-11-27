#!/bin/bash
W="<scratchpad>/indep_review"
cd "$W"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$W/numba_cache_p3" OMP_NUM_THREADS=1
PY=<conda>/envs/skforecast_27_py13_p3/python.exe
for est in ridge lgbm hgb xgb; do
  for sc in float mixed; do
    timeout 1500 $PY perf_time.py $est $sc 300 1000 5 > logs/perf_time_${est}_${sc}.log 2>&1; echo "perf_time $est $sc exit=$?"
  done
done
timeout 900 $PY perf_time.py catboost mixed 300 1000 3 > logs/perf_time_catboost_mixed.log 2>&1; echo "perf_time catboost mixed exit=$?"
timeout 600 ./run.sh branch p3 perf_convert.py > logs/perf_convert.log 2>&1; echo "perf_convert exit=$?"
for est in ridge lgbm xgb hgb; do
  for sc in float mixed; do
    for op in fit search; do
      for tree in base branch base branch; do
        timeout 600 ./run.sh $tree p3 perf_mem.py $est $sc $op 300 1000 >> logs/perf_mem.log 2>&1
      done
    done
  done
done
for tree in base branch base branch; do
  timeout 600 ./run.sh $tree p3 perf_mem.py catboost mixed fit 300 1000 >> logs/perf_mem.log 2>&1
done
echo PERFDONE
