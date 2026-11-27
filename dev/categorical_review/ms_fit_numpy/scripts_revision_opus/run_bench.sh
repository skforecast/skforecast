#!/bin/bash
W="<scratchpad>/indep_review_opus"
cd "$W"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1
PY=<conda>/envs/skforecast_27_py13_p3/python.exe
BR="<repo>"; BA="$W/base"
: > bench_results.jsonl
CFGS="lgbm:float:ordinal lgbm:mixed:ordinal lgbm:float:ordinal_category ridge:float:ordinal ridge:mixed:ordinal hgb:mixed:ordinal xgb:float:ordinal xgb:mixed:ordinal"
for rep in 1 2; do
  for c in $CFGS; do
    IFS=: read est ex enc <<< "$c"
    if [ $rep = 1 ]; then ORDER="base branch"; else ORDER="branch base"; fi
    for v in $ORDER; do
      if [ $v = branch ]; then PP=$BR; else PP=$BA; fi
      out=$(PYTHONPATH="$PP" timeout 600 $PY bench.py $est $ex $enc 3 2>/dev/null | tail -1)
      echo "{\"rep\": $rep, \"variant\": \"$v\", \"est\": \"$est\", \"ex\": \"$ex\", \"enc\": \"$enc\", \"r\": $out}" >> bench_results.jsonl
    done
  done
done
echo BENCH_DONE
