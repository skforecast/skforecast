"""
Prototype of proposal NORI-P2: drop only the context rows whose target is NaN
and keep the rows whose covariates are NaN, which `NoriRegressor` accepts
(synthefy-nori README: "Leave missing values as NaN").

Two modes:

    # Backtesting of the user guide with the current adapter or the patch
    python patch_keep_nan_covariates.py guide --variant current --out ...
    python patch_keep_nan_covariates.py guide --variant patched --out ...

    # Targeted check: context exog with NaN, rows kept by the patch and
    # identical result to a direct NoriRegressor call on the same matrix
    python patch_keep_nan_covariates.py nan-check

Run from the repository root with ~/.venvs/fm-review-nori/bin/python.
"""

import runpy
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel
from skforecast.foundation._adapters import NoriAdapter

ORIGINAL_PREDICT = NoriAdapter.predict
CAPTURED = {}


def patched_predict(self, steps, context, context_exog, exog, quantiles):
    """
    Copy of `NoriAdapter.predict` where only rows with NaN target are dropped.
    """
    quantile_list = list(quantiles) if quantiles is not None else None
    if quantile_list is not None and any((q <= 0.0) or (q >= 1.0) for q in quantile_list):
        raise ValueError("NoriAdapter quantiles must lie strictly in (0, 1).")
    if quantile_list is not None:
        query_levels = sorted(set(quantile_list))
        quantile_column_indices = [query_levels.index(q) for q in quantile_list]
    self._load_model()
    first_series = next(iter(context.values()))
    is_datetime = isinstance(first_series.index, pd.DatetimeIndex)
    predictions = {}
    for series_name, series in context.items():
        ctx_exog = context_exog.get(series_name) if context_exog is not None else None
        fut_exog = exog.get(series_name) if exog is not None else None
        exog_cols = self._known_future_columns(ctx_exog, fut_exog)
        X_ctx = self._featurize(series, is_datetime, ctx_exog, exog_cols, offset=0, n=len(series))
        X_fut = self._featurize(series, is_datetime, fut_exog, exog_cols, offset=len(series), n=steps)
        y_ctx = series.to_numpy(dtype=float)
        valid_rows = ~np.isnan(y_ctx)  # the only change
        if not valid_rows.any():
            raise ValueError(f"Series '{series_name}' has no context rows without NaN.")
        X_ctx = X_ctx[valid_rows]
        y_ctx = y_ctx[valid_rows]
        CAPTURED[series_name] = (X_ctx, y_ctx, X_fut)
        self._model.fit(X_ctx, y_ctx)
        if quantile_list is None:
            y_hat = self._model.predict(X_fut, output_type=self.point_estimate)
            predictions[series_name] = self._to_numpy(y_hat).reshape(-1, 1)
        else:
            q = self._to_numpy(
                self._model.predict(X_fut, output_type="quantiles", quantiles=query_levels)
            )
            q = q.reshape(len(query_levels), steps).T
            predictions[series_name] = q[:, quantile_column_indices]
    return predictions


def guide(argv):
    variant = argv[argv.index("--variant") + 1]
    rest = [a for i, a in enumerate(argv) if a != "--variant" and (i == 0 or argv[i - 1] != "--variant")]
    if variant == "patched":
        NoriAdapter.predict = patched_predict
    print(f"variant: {variant}")
    script = Path(__file__).resolve().parents[1] / "common" / "backtest_guide.py"
    sys.argv = [str(script), "--model-id", "Synthefy/Nori", "--exog", *rest]
    runpy.run_path(str(script), run_name="__main__")


def nan_check():
    warnings.simplefilter("ignore")
    for pkg in ["synthefy-nori", "torch", "numpy", "pandas", "skforecast"]:
        print(f"{pkg:14s}:", version(pkg))
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=224, freq="h")
    temp = pd.Series(np.cos(2 * np.pi * idx.hour / 24) + rng.normal(0, 0.1, len(idx)), index=idx, name="temp")
    y = pd.Series(30 + 4 * np.sin(2 * np.pi * idx.hour / 24) + 3 * temp.to_numpy()
                  + rng.normal(0, 0.3, len(idx)), index=idx, name="y")
    y, y_test = y.iloc[:200], y.iloc[200:]
    exog, exog_fut = temp.iloc[:200].to_frame(), temp.iloc[200:].to_frame()
    exog_nan = exog.copy()
    exog_nan.iloc[::4, 0] = np.nan  # 25 % of the context covariate missing

    results = {}
    for variant in ["current", "patched"]:
        NoriAdapter.predict = ORIGINAL_PREDICT if variant == "current" else patched_predict
        fm = FoundationModel(model_id="Synthefy/Nori", context_length=200)
        fm.fit(series=y, exog=exog_nan)
        rows_used = {}
        model_fit = None
        fm.adapter._load_model()
        model_fit = fm.adapter._model.fit

        def spy(X, yy, _fit=model_fit):
            rows_used["n"] = len(yy)
            return _fit(X, yy)

        fm.adapter._model.fit = spy
        pred = fm.predict(steps=24, exog=exog_fut)["pred"].to_numpy(dtype=float)
        results[variant] = pred
        mae = np.mean(np.abs(pred - y_test.to_numpy()))
        print(f"{variant:8s}: context rows passed to NoriRegressor.fit = {rows_used['n']} of 200, "
              f"MAE on the 24 hidden steps = {mae:.4f} (indicative, one window)")

    from synthefy_nori import NoriRegressor
    X_ctx, y_ctx, X_fut = CAPTURED["y"]
    direct = NoriRegressor(model="Synthefy/Nori").fit(X_ctx, y_ctx).predict(X_fut)
    print(f"patched vs direct NoriRegressor on the same matrix with NaN: max diff "
          f"{np.max(np.abs(direct - results['patched'])):.3e}")


if __name__ == "__main__":
    if sys.argv[1] == "guide":
        guide(sys.argv[2:])
    else:
        nan_check()
