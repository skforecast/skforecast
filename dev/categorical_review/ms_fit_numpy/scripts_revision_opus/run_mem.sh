#!/bin/bash
W="<scratchpad>/indep_review_opus"
cd "$W"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1
PY=<conda>/envs/skforecast_27_py13_p3/python.exe
BR="<repo>"; BA="$W/base"
: > mem_results.jsonl
for c in lgbm:float:ordinal lgbm:mixed:ordinal lgbm:float:ordinal_category ridge:mixed:ordinal xgb:mixed:ordinal; do
  IFS=: read est ex enc <<< "$c"
  for op in fit osa; do
    for v in base branch base branch; do
      if [ $v = branch ]; then PP=$BR; else PP=$BA; fi
      out=$(PYTHONPATH="$PP" timeout 600 $PY bench_mem.py $est $ex $enc 0 $op 2>/dev/null | tail -1)
      echo "{\"variant\": \"$v\", \"cfg\": \"$c\", \"r\": $out}" >> mem_results.jsonl
    done
  done
done
echo MEM_DONE
