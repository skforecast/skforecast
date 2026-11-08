#!/usr/bin/env bash
# Work threshold of n_jobs='auto' on macOS or Linux (section 16 of the report).
#
# 1. Machine information (results/<tag>/machine.json).
# 2. Unit tests of the threshold and of the searches (results/<tag>/tests.log).
# 3. Block A: cold and warm start of k workers (bench_spawn.py).
# 4. V1: single backtestings in a fresh process with n_jobs=1, 'auto' and
#    cpu_count() - 1 (bench_cold.py).
# 5. Analysis: suggested threshold and check of 'auto' (threshold.py).
#
# Run it from any folder, with the conda environment of skforecast active and
# the machine plugged in, idle and without other heavy programs (about 40 min
# with 5 repetitions). It tests and measures the skforecast of this
# repository (PYTHONPATH), not the one installed in the environment.
#
# Usage: bash run_platform.sh <tag> [repetitions]
#   tag: name of the results folder, for example macos_m3_pro
#   PYTHON: interpreter to use (default: python)

set -u
TAG="${1:?results tag, for example macos_m3_pro}"
REPS="${2:-5}"
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
if not hasattr(_utils, 'adjust_n_jobs_to_work'):
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
  "$ROOT/skforecast/model_selection/tests/tests_utils/test_adjust_n_jobs_to_work.py" \
  "$ROOT/skforecast/model_selection/tests/tests_validation/test_backtesting_forecaster.py" \
  "$ROOT/skforecast/model_selection/tests/tests_validation/test_backtesting_forecaster_multiseries.py" \
  "$ROOT/skforecast/model_selection/tests/tests_search/test_grid_search_forecaster.py" \
  "$ROOT/skforecast/model_selection/tests/tests_search/test_grid_search_forecaster_multiseries.py" \
  -k "adjust_n_jobs_to_work or adjusted_to_the_work or resolved_with_the_rules" \
  > "$NJOBS_RESULTS_DIR/tests.log" 2>&1
rc=$?
tail -1 "$NJOBS_RESULTS_DIR/tests.log"
done_ tests $rc

step A
"$PY" bench_spawn.py > "$NJOBS_RESULTS_DIR/bench_spawn.log" 2>&1
done_ A $?

step V1
"$PY" bench_cold.py --reps "$REPS" --values 1,auto,cpu-1 \
  > "$NJOBS_RESULTS_DIR/bench_cold.log" 2>&1
done_ V1 $?

step threshold
"$PY" threshold.py --results-dir "$NJOBS_RESULTS_DIR"
done_ threshold $?

echo "ALL DONE $(( ($(date +%s) - T0) / 60 )) min. Results in $NJOBS_RESULTS_DIR"
