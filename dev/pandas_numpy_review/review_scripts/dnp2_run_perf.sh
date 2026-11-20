#!/usr/bin/env bash
# D-NP2: `dnp2_time_fit.py` for the current code and for the prototype, one
# process per variant, one after another. Run it alone on the machine.
# Usage: TODAY=<0.28.x checkout> PROTO=<0.28.x + dnp2_proto.patch> \
#        REAL_DATA_DIR=<dir of real_data.py> bash dnp2_run_perf.sh ENV OUT [SPECS]
# ENV: p2, p3 (pandas 3 without pyarrow) or p3pa. SPECS: "case:estimator ...".
# DNP2_CATBOOST_DF=1 measures the prototype variant that keeps a DataFrame
# for CatBoost with categorical features.
D=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=1
case $1 in
  p2) PY=~/.venvs/skforecast-p2/bin/python; export NO_PYARROW=0;;
  p3) PY=~/.venvs/skforecast/bin/python; export NO_PYARROW=1;;
  p3pa) PY=~/.venvs/skforecast/bin/python; export NO_PYARROW=0;;
  *) echo "ENV must be p2, p3 or p3pa"; exit 1;;
esac
OUT=$2
SPECS=${3:-"float:lgbm float:xgb float:hgb float:ridge float:catboost ordinal_category:lgbm ordinal_category:hgb int10:lgbm int10:ridge text2:lgbm text2:hgb text2:catboost none_cat2:lgbm none_cat2:ridge bdg2:lgbm"}
: > "$OUT"
for spec in $SPECS; do
  for tree in "$TODAY" "$PROTO"; do
    PYTHONPATH=$tree:$D $PY "$D/dnp2_time_fit.py" "${spec%%:*}" "${spec##*:}" "${REPS:-3}" 2>&1 | tail -1 >> "$OUT"
  done
done
PYTHONPATH=$TODAY:$D $PY -c "import _env, lightgbm, xgboost, catboost; print(_env.versions())" >> "$OUT"
