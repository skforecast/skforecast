# Unit test backtesting_forecaster CatBoost refit
# ==============================================================================
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
import numpy as np


def test_output_backtesting_forecaster_refit_CatBoost_with_categorical_features():
    """
    Test that a backtesting with refit of a ForecasterRecursive with a CatBoost
    estimator and a categorical exogenous variable returns the predictions of a
    new forecaster fitted in each fold, with one job and with two jobs. With
    categorical features, CatBoost does not give the model of a new object
    when a fitted model is fitted again, and `predict` can block forever with a
    copy of a fitted model, which is what the folds of a parallel backtesting
    receive. The backtesting runs in a subprocess with a time limit, and the
    two jobs are threads, which also receive a copy of the forecaster, so that
    no worker process is left running.
    """
    code = textwrap.dedent("""
        import json
        import joblib
        import numpy as np
        import pandas as pd
        from catboost import CatBoostRegressor
        from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
        from skforecast.recursive import ForecasterRecursive

        rng = np.random.RandomState(7)
        n = 240
        index = pd.date_range("2020-01-01", periods=n, freq="D")
        codes = rng.permutation(np.arange(n) % 6)
        effect = np.array([5.0, -3.0, 9.0, 0.0, -8.0, 2.0])
        y = pd.Series(
            effect[codes] + rng.normal(scale=0.3, size=n), index=index, name="y"
        )
        exog = pd.DataFrame(
            {"c": pd.Categorical(codes), "x": rng.normal(size=n)}, index=index
        )
        cv = TimeSeriesFold(
            steps=10, initial_train_size=200, refit=True, fixed_train_size=True
        )

        def create_forecaster():
            return ForecasterRecursive(
                estimator=CatBoostRegressor(
                    iterations=30, depth=3, verbose=0, thread_count=1,
                    random_seed=1, allow_writing_files=False
                ),
                lags=3
            )

        def backtesting_predictions(n_jobs):
            _, predictions = backtesting_forecaster(
                forecaster=create_forecaster(), y=y, exog=exog, cv=cv,
                metric="mean_absolute_error", n_jobs=n_jobs, verbose=False,
                show_progress=False
            )
            return predictions["pred"].to_list()

        predictions = {"new": []}
        for start in range(200, n, 10):
            forecaster = create_forecaster()
            forecaster.fit(
                y=y.iloc[start - 200:start], exog=exog.iloc[start - 200:start]
            )
            predictions["new"].extend(
                forecaster.predict(steps=10, exog=exog.iloc[start:start + 10])
            )
        predictions["n_jobs_1"] = backtesting_predictions(n_jobs=1)
        with joblib.parallel_config(backend="threading"):
            predictions["n_jobs_2"] = backtesting_predictions(n_jobs=2)
        print(json.dumps(predictions))
    """)
    repo_root = Path(__file__).resolve().parents[4]

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        encoding       = "utf-8",
        errors         = "replace",
        cwd            = repo_root,
        env            = {**os.environ, "PYTHONPATH": str(repo_root)},
        timeout        = 180
    )

    assert result.returncode == 0, result.stderr
    predictions = json.loads(result.stdout.splitlines()[-1])

    # NOTE: CatBoost results with categorical features change between platforms
    # (Linux, macOS), so the expected values are not hardcoded. They are the
    # predictions of a new forecaster fitted in each fold in the subprocess.
    expected = np.array(predictions["new"])

    assert expected.shape == (40,)
    np.testing.assert_array_almost_equal(predictions["n_jobs_1"], expected)
    np.testing.assert_array_almost_equal(predictions["n_jobs_2"], expected)
