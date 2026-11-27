#!/bin/bash
# Usage: run_tests.sh <tag> <tree>  runs the multiseries tests of <tree> with that tree
D="<scratchpad>/doubts_b"
PY="<conda>/envs/skforecast_27_py13_p3/python.exe"
tag=$1
tree=$2
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$tree"
cd "$tree"
"$PY" -c "import skforecast; print(skforecast.__file__)" > "$D/tests_${tag}.txt" 2>&1
timeout 3000 "$PY" -m pytest -q -p no:cacheprovider -rf \
  skforecast/recursive/tests/tests_forecaster_recursive_multiseries \
  skforecast/model_selection/tests/tests_utils/test_predict_and_calculate_metrics_one_step_ahead_multiseries.py \
  skforecast/model_selection/tests/tests_utils/test_train_test_split_one_step_ahead_search.py \
  >> "$D/tests_${tag}.txt" 2>&1
echo "$tag tests exit=$?"
