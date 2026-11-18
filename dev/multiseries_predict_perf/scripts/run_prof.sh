#!/usr/bin/env bash
# Line profile of one case in the three environments. One JSON per environment
# in $MS_WORK/out/<tag>.
#
# Usage: run_prof.sh <tag> <tree: base|new> <operation> <n_series> <exog_kind> [fmt] [steps]
#   operation: predict | predict_stored | backtesting | fit | create_train_X_y |
#              predict_mv | predict_mv_stored
# With EXOG_DTYPES=mixed the exog has an `int` and a `category` column.
# With PIN=1 the process is pinned (for the timed profiles); without it only
# the number of times every line runs is reliable.
# Then: python linediff.py $MS_WORK/out/<tag> <case>      (pandas 2 vs pandas 3)
#       python protodiff.py <dir of base> <dir of new> <case>   (base vs new)
source "$(dirname "$0")/env.sh"
TAG="$1"; select_tree "$2"; shift 2
mkdir -p "$OUT/$TAG"
cd "$SCRIPTS" || exit 1
export PYTHONPATH="$TREE_DIR" PYTHONPYCACHEPREFIX="$MS_WORK/pycache_prof"
if [ "$PIN" = "1" ]; then export BENCH_CPU="${MS_CPU:-12}"; fi
NAME=$(echo "$*" | tr ' ' '_')
for env in ${ENVS:-p2 p3 p3nopa}; do
  select_env "$env"
  NO_PYARROW=$NOPA "$PY" lineprof.py "$OUT/$TAG/${NAME}__${env}.json" "$@" 2>&1 | tail -1
done
