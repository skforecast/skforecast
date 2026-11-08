#!/usr/bin/env bash
# Work threshold of n_jobs='auto' on macOS or Linux (section 16 of the report).
#
# 1. Machine information (results/<tag>/machine.json).
# 2. Unit tests of the threshold and of the searches (results/<tag>/tests.log).
# 3. Block A: cold and warm start of k workers (bench_spawn.py). The cold
#    start of cpu_count() - 1 workers is what sets the threshold.
# 4. With --full only: V1, single backtestings in a fresh process with
#    n_jobs=1, 'auto' and cpu_count() - 1 (bench_cold.py), to confirm the
#    threshold with the full method and check 'auto' case by case.
# 5. Analysis: suggested threshold and, with --full, check of 'auto'
#    (threshold.py).
#
# Run it from any folder, with the conda environment of skforecast active and
# the machine plugged in, idle and without other heavy programs. About 6 min;
# with --full, about 40 min (5 repetitions). It tests and measures the
# skforecast of this repository (PYTHONPATH), not the one installed in the
# environment.
#
# Usage: bash run_platform.sh <tag> [--full] [--reps N]
#   tag: name of the results folder, for example macos_m3_pro
#   --full: also run V1 (N repetitions, default 5)
#   PYTHON: interpreter to use (default: python)

set -u
TAG="${1:?results tag, for example macos_m3_pro}"
shift
FULL=0
REPS=5
while [ $# -gt 0 ]; do
  case "$1" in
    --full) FULL=1 ;;
    --reps) shift; REPS="${1:?--reps needs a number}" ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python}"

cd "$HERE"
export PYTHONPATH="$ROOT"
export PYTHONIOENCODING=utf-8
export NJOBS_REPO_ROOT="$ROOT"
export NJOBS_RESULTS_DIR="$HERE/results/$TAG"
mkdir -p "$NJOBS_RESULTS_DIR"

step() { echo "STEP $1 START $(date +%T)"; }
done_() { echo "STEP $1 END rc=$2 $(date +%T)"; }
T0=$(date +%s)

step machine
"$PY" -c "
from common import RESULTS_DIR, check_repo_skforecast, dump_json, machine_info
check_repo_skforecast()
import skforecast
from skforecast.model_selection import _utils
print('skforecast', skforecast.__version__, skforecast.__file__)
if not hasattr(_utils, 'limit_n_jobs_by_workload'):
    raise SystemExit('This skforecast has no work threshold: check out its branch.')
try:
    dump_json(machine_info(), RESULTS_DIR / 'machine.json')
except Exception as exc:
    print(f'WARNING: machine information not saved: {exc!r}')
"
rc=$?
done_ machine $rc
[ $rc -eq 0 ] || exit $rc

step tests
"$PY" -m pytest -q -p no:cacheprovider \
  "$ROOT/skforecast/model_selection/tests/tests_utils/test_limit_n_jobs_by_workload.py" \
  "$ROOT/skforecast/model_selection/tests/tests_validation/test_backtesting_forecaster.py" \
  "$ROOT/skforecast/model_selection/tests/tests_validation/test_backtesting_forecaster_multiseries.py" \
  "$ROOT/skforecast/model_selection/tests/tests_search/test_grid_search_forecaster.py" \
  "$ROOT/skforecast/model_selection/tests/tests_search/test_grid_search_forecaster_multiseries.py" \
  -k "limit_n_jobs_by_workload or limited_by_the_workload or resolved_with_the_rules" \
  > "$NJOBS_RESULTS_DIR/tests.log" 2>&1
rc=$?
tail -1 "$NJOBS_RESULTS_DIR/tests.log"
done_ tests $rc

step A
"$PY" bench_spawn.py > "$NJOBS_RESULTS_DIR/bench_spawn.log" 2>&1
done_ A $?

if [ "$FULL" -eq 1 ]; then
  step V1
  "$PY" bench_cold.py --reps "$REPS" --values 1,auto,cpu-1 \
    > "$NJOBS_RESULTS_DIR/bench_cold.log" 2>&1
  done_ V1 $?
fi

step threshold
"$PY" threshold.py --results-dir "$NJOBS_RESULTS_DIR"
done_ threshold $?

echo "ALL DONE $(( ($(date +%s) - T0) / 60 )) min. Results in $NJOBS_RESULTS_DIR"
