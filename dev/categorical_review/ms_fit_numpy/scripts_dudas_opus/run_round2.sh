#!/bin/bash
# Round 2: branch tests on minimal_bt (with interface shim), base tests on
# minimal v2, scenario matrix on minimal v2.
D="<scratchpad>/doubts_b"
PY="<conda>/envs/skforecast_27_py13_p3/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
TESTS="skforecast/recursive/tests/tests_forecaster_recursive_multiseries skforecast/model_selection/tests/tests_utils/test_predict_and_calculate_metrics_one_step_ahead_multiseries.py skforecast/model_selection/tests/tests_utils/test_train_test_split_one_step_ahead_search.py"

cd "$D/minimal_bt"
export PYTHONPATH="$D/minimal_bt"
timeout 3000 "$PY" -m pytest -q -p no:cacheprovider --tb=line -rf $TESTS > "$D/tests_minimal_bt_shim.txt" 2>&1
echo "minimal_bt_shim exit=$?"

cd "$D/minimal"
export PYTHONPATH="$D/minimal"
timeout 3000 "$PY" -m pytest -q -p no:cacheprovider --tb=line -rf $TESTS skforecast/model_selection/tests/tests_search > "$D/tests_minimal_basetests.txt" 2>&1
echo "minimal_basetests exit=$?"

bash "$D/run_scenarios.sh" minimal2 "$D/minimal"
