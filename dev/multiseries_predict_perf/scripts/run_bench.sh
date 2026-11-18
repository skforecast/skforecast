#!/usr/bin/env bash
# Timings in alternated processes: round -> environment -> tree. One JSON per
# (round, environment, tree) in $MS_WORK/out/<tag>. Every process is pinned to
# one logical processor with high priority.
#
# Usage: run_bench.sh <tag> <rounds> "<trees: base new>" [cases]
#   cases: a set of bench_ms.py (@core, @adverse, @scale) or a regex of case
#   names; all the cases if omitted.
# Environments: p2 p3 p3nopa, or the ones in ENVS.
# Then: python agg_ms.py $MS_WORK/out/<tag> [--gain]
source "$(dirname "$0")/env.sh"
TAG="$1"; ROUNDS="$2"; TREES="$3"; CASES="$4"
mkdir -p "$OUT/$TAG"
cd "$SCRIPTS" || exit 1
export BENCH_CPU="${MS_CPU:-12}"
for r in $(seq 1 "$ROUNDS"); do
  for env in ${ENVS:-p2 p3 p3nopa}; do
    select_env "$env"
    for tree in $TREES; do
      select_tree "$tree"
      PYTHONPATH="$TREE_DIR" PYTHONPYCACHEPREFIX="$MS_WORK/pycache_$tree" NO_PYARROW=$NOPA \
        "$PY" bench_ms.py "$OUT/$TAG/r${r}_${env}_${tree}.json" ${CASES:+"$CASES"} 2>&1 | tail -1
    done
  done
done
echo DONE "$TAG"
