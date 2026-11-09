"""
H6: threads used by each estimator inside a loky worker (n_jobs workers) and in
the main process. Prints one line per (place, estimator).
"""

import json
import os
import sys
import warnings

from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


def probe(place):
    import numpy as np
    from joblib import effective_n_jobs
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = X[:, 0]
    out = {"place": place, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
    out["hgb_openmp_threads"] = _openmp_effective_n_threads()
    from lightgbm import LGBMRegressor
    for nj in (None, -1):
        m = LGBMRegressor(n_jobs=nj, verbose=-1).fit(X, y)
        out[f"lgbm_n_jobs={nj}"] = m.booster_.params.get("num_threads")
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=5).fit(X, y)
    cfg = json.loads(m.get_booster().save_config())
    out["xgb_n_jobs=None_nthread"] = cfg["learner"]["generic_param"]["nthread"]
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=5, verbose=0, allow_writing_files=False).fit(X, y)
    out["cat_thread_count=-1"] = m.get_all_params().get("thread_count")
    out["joblib_effective_n_jobs(-1)_nested"] = effective_n_jobs(-1)
    return out


n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
print(json.dumps(probe("main")))
for r in Parallel(n_jobs=n_jobs)(delayed(probe)(f"worker of {n_jobs}") for _ in range(1)):
    print(json.dumps(r))
