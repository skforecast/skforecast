"""WSL copy only: emulate the D9 fix and drop the slow block C cells.

- skforecast/utils/utils.py: the LightGBM fast path passes the same
  num_threads as LGBMRegressor.predict (derived from n_jobs).
- dev/n_jobs_review/bench.py: reduced block C without lgbm_none and xgb_none
  (347 s per sequential call and a timeout in the pilot, not needed for D4).
"""

import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in pairs:
        assert text.count(old) == 1, (path.name, old)
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched {path.relative_to(root)}")


patch(
    root / "skforecast" / "utils" / "utils.py",
    [
        (
            "        booster = estimator.booster_\n\n"
            "        def predict_fn(X):\n"
            "            return booster.predict(X)\n",
            "        booster = estimator.booster_\n"
            "        num_threads = estimator._process_n_jobs(estimator.n_jobs)\n\n"
            "        def predict_fn(X):\n"
            "            return booster.predict(X, num_threads=num_threads)\n",
        )
    ],
)
patch(
    root / "dev" / "n_jobs_review" / "bench.py",
    [
        (
            '        ests = ["lgbm_none", "xgb_none", "rf_m1", "hgb"]',
            '        ests = ["rf_m1", "hgb"]',
        )
    ],
)
