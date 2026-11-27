#!/bin/bash
W="<scratchpad>/indep_review"
REPO="<repo>"
cd "$W"
# 1. alternative prototype: matrix, probes, user configuration (pandas 3)
for est in lgbm xgb hgb ridge catboost ttr_lgbm pipe_lgbm; do
  for mode in wide dict; do
    timeout 900 ./run.sh alt p3 matrix.py $est out/m_alt_p3_${est}_${mode}.pkl $mode > logs/m_alt_p3_${est}_${mode}.log 2>&1
    echo "alt matrix $est $mode exit=$?"
  done
done
timeout 900 ./run.sh alt p3 probes.py P01,P02,P10 > logs/probes_alt_p3.txt 2>&1; echo "alt probes exit=$?"
timeout 1200 ./run.sh alt p3 user_cat_config.py > logs/usercat_alt_p3.txt 2>&1; echo "alt usercat exit=$?"
timeout 600 ./run.sh base p3 spy_consistency.py > logs/spy_base_p3.txt 2>&1; echo "spy base exit=$?"
timeout 600 ./run.sh branch p3 spy_consistency.py > logs/spy_branch_p3.txt 2>&1; echo "spy branch exit=$?"
# 2. tests
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
T1="skforecast/recursive/tests/tests_forecaster_recursive_multiseries"
T2="skforecast/utils/tests/tests_utils/test_resolve_user_categorical_config.py skforecast/utils/tests/tests_utils/test_train_matrix_to_estimator_input.py skforecast/utils/tests/tests_utils/test_check_in_place_fit.py skforecast/utils/tests/tests_utils/test_check_exog_dtypes.py skforecast/model_selection/tests/tests_utils/test_predict_and_calculate_metrics_one_step_ahead_multiseries.py skforecast/model_selection/tests/tests_utils/test_train_test_split_one_step_ahead_search.py skforecast/direct/tests/tests_forecaster_direct/test_create_train_X_y.py skforecast/direct/tests/tests_forecaster_direct_multivariate/test_create_train_X_y.py skforecast/recursive/tests/tests_forecaster_recursive/test_create_train_X_y.py skforecast/recursive/tests/tests_forecaster_recursive_classifier/test_create_train_X_y.py"
T3=$(cd "$REPO" && ls skforecast/model_selection/tests/tests_search/*multiseries*.py | tr '\n' ' ')
for pd in p3 p2; do
  if [ "$pd" = "p2" ]; then ENV=skforecast_27_py13; else ENV=skforecast_27_py13_p3; fi
  PY=<conda>/envs/$ENV/python.exe
  export NUMBA_CACHE_DIR="$W/numba_cache_$pd"
  cd "$REPO"
  $PY -m pytest $T1 -q -p no:cacheprovider > "$W/logs/pytest_branch_${pd}_multiseries.log" 2>&1; echo "pytest branch $pd multiseries exit=$?"
  $PY -m pytest $T2 -q -p no:cacheprovider > "$W/logs/pytest_branch_${pd}_changed.log" 2>&1; echo "pytest branch $pd changed exit=$?"
  $PY -m pytest $T3 -q -p no:cacheprovider > "$W/logs/pytest_branch_${pd}_search.log" 2>&1; echo "pytest branch $pd search exit=$?"
done
# new and modified tests against the base code
cd "$W/base_newtests"
export NUMBA_CACHE_DIR="$W/numba_cache_p3"
<conda>/envs/skforecast_27_py13_p3/python.exe -m pytest $T1/test_fit.py $T1/test_predict.py $T1/test_set_in_sample_residuals.py $T1/test_create_predict_X.py skforecast/model_selection/tests/tests_utils/test_predict_and_calculate_metrics_one_step_ahead_multiseries.py skforecast/utils/tests/tests_utils/test_check_in_place_fit.py -q -p no:cacheprovider -rf > "$W/logs/pytest_base_newtests_p3.log" 2>&1; echo "pytest base+newtests exit=$?"
cd "$W"
echo CHAIN3DONE
