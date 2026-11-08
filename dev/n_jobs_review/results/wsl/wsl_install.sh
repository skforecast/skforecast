#!/usr/bin/env bash
# Install the same package versions as the Windows dev env (skforecast_27_py13).
set -u
P="$HOME/miniconda3/envs/skforecast_27_py13_wsl/bin/python"
"$P" -m pip install -q \
  numpy==2.5.4 pandas==2.3.3 scikit-learn==1.9.1 scipy==1.18.1 joblib==1.6.0 \
  lightgbm==4.7.0 xgboost==3.4.1 catboost==1.2.10 statsmodels==0.15.0 \
  numba==0.68.0 llvmlite==0.50.0 psutil==7.2.2 threadpoolctl==3.7.0 \
  optuna==5.0.0 tqdm==4.70.1 rich==15.0.0 cloudpickle==3.1.2 pyarrow==26.0.0 \
  2>&1 | tail -5
echo "pip exit=${PIPESTATUS[0]}"
"$P" -m pip list 2>/dev/null | grep -i -E "^(numpy|pandas|scikit-learn|scipy|joblib|lightgbm|xgboost|catboost|statsmodels|numba|psutil|threadpoolctl|optuna) "
