#!/usr/bin/env bash
# Run a review script in the three pandas environments, one after another.
# Usage: bash run3.sh script.py [args...]
# SK_TREE: checkout of perf/multiseries-predict-backtesting (ab62227).
D=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=1
export PYTHONPATH="${SK_TREE:-/home/user/sk_perf}:$D"
P3=~/.venvs/skforecast/bin/python
P2=~/.venvs/skforecast-p2/bin/python
echo "=== p2 ===";      $P2 "$@"
echo "=== p3 (sin pyarrow) ==="; NO_PYARROW=1 $P3 "$@"
echo "=== p3 pa ===";   $P3 "$@"
