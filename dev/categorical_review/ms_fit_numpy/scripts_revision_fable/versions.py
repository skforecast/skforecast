import sys, importlib
print(sys.version)
for m in ["numpy","pandas","sklearn","lightgbm","xgboost","catboost","pyarrow","scipy","numba","optuna","skops","joblib"]:
    try:
        mod = importlib.import_module(m)
        print(m, mod.__version__)
    except Exception as e:
        print(m, "NOT INSTALLED", type(e).__name__)
import skforecast
print(skforecast.__file__, skforecast.__version__)
