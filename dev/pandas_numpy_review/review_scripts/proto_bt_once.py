"""
Prototype of a multi-series backtesting without refit that validates and
prepares once, loops over the folds with numpy only and builds the output once.

Unlike `scripts/ceiling_bt_ms.py` (the ceiling of the study), it returns the
same two DataFrames as `backtesting_forecaster_multiseries` (metrics and
predictions, with `fold`, `step`, `y_true`), supports series of different
length, NaNs, levels that are not predicted in every fold, `gap`,
`transformer_series`, differentiation, window and calendar features and every
`encoding`, and can emit the per-fold warnings of `check_predict_input` from
masks computed once.

Scope (anything else must go through the current path): `refit=False`,
`initial_train_size` not None, `interval=None`, `return_predictors=False`,
`n_jobs=1`, `ForecasterRecursiveMultiSeries`, a single differentiation order,
exog as dict of DataFrames or wide DataFrame aligned with `span_index`.

Nothing in `skforecast/` is modified: private functions are called as they are.
"""
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from skforecast.exceptions import MissingExogWarning, MissingValuesWarning
from skforecast.metrics import _get_metric, add_y_train_argument
from skforecast.model_selection._utils import (
    _calculate_metrics_backtesting_multiseries,
    _extract_data_folds_multiseries,
    _initialize_levels_model_selection_multiseries,
    _warn_levels_not_predicted_multiseries,
)
from skforecast.utils import deepcopy_forecaster
from skforecast.utils.utils import (
    check_preprocess_exog_multiseries,
    check_preprocess_series,
    estimator_has_native_nan_support,
    transform_dataframe,
    transform_numpy,
)


class _Window:
    """Minimal stand-in for the DataFrame `_recursive_predict` receives."""

    __slots__ = ("values",)

    def __init__(self, values):
        self.values = values

    def to_numpy(self):
        return self.values


def _warn_exog_fold(
    names, cols, exog, exog_state, exog_obj_of_col, raw_isna, raw_rows, n_obj,
    name_to_col, span_index, test_start, test_end,
):
    """Warnings of `check_predict_input` about `exog` in one fold, from masks."""
    steps = test_end - test_start
    n_rows_obj = raw_rows[:, test_start:test_end].sum(axis=1)
    objs = exog_obj_of_col[cols]
    state = exog_state[cols]
    is_missing = (state == 2) | ((state == 1) & (n_rows_obj[objs] == 0))
    if is_missing.any():
        # NOTE: Built as in `check_predict_input` (`set(levels) - set(keys)`),
        # so that the text of the set in the message is the same.
        keys = [names[c] for c in cols[~is_missing]]
        no_exog = set(names[c] for c in cols) - set(keys)
        warnings.warn(
            f"`exog` does not contain keys for levels {no_exog}. "
            f"Missing levels are filled with NaN. Most of machine learning "
            f"models do not allow missing values. Prediction method may fail.",
            MissingExogWarning,
        )
    # Objects with something to warn about, all at once: usually none.
    has_nan = raw_isna[:, test_start:test_end].any(axis=1)
    short = raw_rows[:, test_start:test_end].sum(axis=1) < steps
    used = np.unique(objs[(state == 1) & ~is_missing])
    if not (has_nan[used] | short[used]).any():
        return
    cols_set = set(cols.tolist())
    items = [  # order of `exog.items()`, as today
        (f"`exog` for series '{name}'", exog_obj_of_col[name_to_col[name]])
        for name in exog
        if name in name_to_col and name_to_col[name] in cols_set
        and exog_state[name_to_col[name]] == 1
        and n_rows_obj[exog_obj_of_col[name_to_col[name]]] > 0
    ]
    for exog_name, obj in items:
        if not (has_nan[obj] or short[obj]):
            continue
        rows = raw_rows[obj, test_start:test_end]
        n_rows = int(rows.sum())
        if raw_isna[obj, test_start:test_end].any():
            warnings.warn(
                f"{exog_name} has missing values. Most of machine learning models "
                f"do not allow missing values. Prediction method may fail.",
                MissingValuesWarning,
            )
        if n_rows < steps:
            warnings.warn(
                f"{exog_name} doesn't have as many values as steps "
                f"predicted, {steps}. Missing values are filled "
                f"with NaN. Most of machine learning models do not "
                f"allow missing values. Prediction method may fail.",
                MissingValuesWarning,
            )
        is_misaligned = ~rows[:n_rows]
        if is_misaligned.any():
            position = int(np.flatnonzero(is_misaligned)[0])
            warnings.warn(
                f"{exog_name} has no value for some of the {steps} steps "
                f"predicted. The first one is {span_index[test_start + position]} "
                f"(position {position}). Missing values are filled with NaN. "
                f"Most of machine learning models do not allow missing values. "
                f"Prediction method may fail.",
                MissingValuesWarning,
            )


def backtesting_once(
    forecaster,
    series,
    cv,
    metric,
    levels=None,
    exog=None,
    add_aggregated_metric=True,
    emit_fold_warnings=True,
    suppress_warnings=False,
    timings=None,
):
    """
    Same result as `backtesting_forecaster_multiseries(..., n_jobs=1,
    show_progress=False)` within the scope of the module docstring.
    `timings`, if a dict, receives the seconds of each stage.
    """
    tic = time.perf_counter
    t0 = tic()
    # As `backtesting_forecaster_multiseries`: series and exog become dicts
    # (a wide exog is the same object for every series).
    series, series_indexes = check_preprocess_series(series)
    if exog is not None:
        series_names_in_ = list(series.keys())
        exog, _ = check_preprocess_exog_multiseries(
            series_names_in_=series_names_in_,
            series_index_type=type(series_indexes[series_names_in_[0]]),
            exog=exog,
            exog_dict={serie: None for serie in series_names_in_},
        )
    t_pre = tic()
    forecaster = deepcopy_forecaster(forecaster, include_out_sample_residuals=False)
    cv = deepcopy(cv)
    cv.set_params(
        {"window_size": forecaster.window_size, "return_all_indexes": False}
    )
    assert cv.refit in (False, 0) and cv.initial_train_size is not None
    levels = _initialize_levels_model_selection_multiseries(
        forecaster=forecaster, series=series, levels=levels
    )
    metrics = [
        _get_metric(metric=m) if isinstance(m, str) else add_y_train_argument(m)
        for m in (metric if isinstance(metric, list) else [metric])
    ]
    forecaster._probabilistic_mode = False
    forecaster._allow_nan_predictions = True
    folds = cv.split(X=series, as_pandas=False)
    span_index = cv._extract_index(X=series)
    gap = cv.gap
    window_size = forecaster.window_size
    dropna_last_window = (
        forecaster.differentiation_max is not None
        or not estimator_has_native_nan_support(forecaster.estimator)
    )

    # ------------------------------------------------------------------
    # Training set and fit: exactly the current code, only for the first fold.
    # ------------------------------------------------------------------
    s_train, _, levels_lw0, e_train, _, _ = next(
        _extract_data_folds_multiseries(
            series=series, folds=folds[:1], span_index=span_index,
            window_size=window_size, exog=exog,
            dropna_last_window=dropna_last_window, externally_fitted=False,
        )
    )
    levels_train = list(s_train)
    t1 = tic()
    forecaster.fit(
        series=s_train, exog=e_train, store_last_window=levels_lw0,
        store_in_sample_residuals=True, suppress_warnings=suppress_warnings,
    )
    t2 = tic()

    # ------------------------------------------------------------------
    # Preparation, once: matrix of series, coverage mask, exog encoded.
    # ------------------------------------------------------------------
    names = list(series.keys()) if isinstance(series, dict) else list(series.columns)
    n_span, n_series = len(span_index), len(names)
    if isinstance(series, dict):
        Y = np.full((n_span, n_series), np.nan)
        covered = np.zeros((n_span, n_series), dtype=bool)
        for j, name in enumerate(names):
            v = series[name]
            if isinstance(v, pd.DataFrame):
                v = v.iloc[:, 0]
            pos = span_index.get_indexer(v.index)
            Y[pos, j] = v.to_numpy(dtype=float)
            covered[pos, j] = True
    else:
        Y = series.to_numpy(dtype=float)
        covered = np.ones((n_span, n_series), dtype=bool)

    name_to_col = {name: j for j, name in enumerate(names)}
    level_cols = np.array([name_to_col[level] for level in levels])
    in_train = np.isin(np.array(names, dtype=object), np.array(levels_train, dtype=object))

    # Series transformed once (the transformers do not change without refit).
    Yt = Y
    has_transformer = any(
        forecaster.transformer_series_.get(name) is not None for name in levels_train
    )
    if has_transformer:
        Yt = Y.copy()
        for j, name in enumerate(names):
            transformer = forecaster.transformer_series_.get(
                name, forecaster.transformer_series_["_unknown_level"]
            )
            if transformer is not None and in_train[j]:
                Yt[:, j] = transform_numpy(
                    array=Y[:, j], transformer=transformer, fit=False,
                    inverse_transform=False,
                )

    order = forecaster.differentiation
    assert order is None or isinstance(order, (int, np.integer))

    # Exog: one row block per distinct object, aligned with `span_index`.
    # exog_state: 0 = key with None (NaN, no warning), 1 = usable object,
    # 2 = key missing or without rows in the training set (`MissingExogWarning`).
    E = None
    if exog is not None:
        exog_names = forecaster.exog_names_in_
        # NOTE: Only the rows of the test sets are aligned and encoded: with a
        # long history and a few folds, encoding the whole exog would cost
        # more than the folds.
        e0 = min(fold[3][0] for fold in folds)
        e1 = max(fold[3][1] for fold in folds)
        frames, raw_isna, raw_rows, obj_of_id = [], [], [], {}
        exog_obj_of_col = np.full(n_series, -1)
        exog_state = np.zeros(n_series, dtype=int)

        def add_object(v):
            if isinstance(v, pd.Series):
                v = v.to_frame()
            pos = span_index.get_indexer(v.index)
            rows = np.zeros(n_span, dtype=bool)
            rows[pos[pos >= 0]] = True
            isna = np.zeros(n_span, dtype=bool)
            isna[pos[pos >= 0]] = v.isna().to_numpy().any(axis=1)[pos >= 0]
            frames.append(v.reindex(index=span_index[e0:e1], columns=exog_names))
            raw_isna.append(isna)
            raw_rows.append(rows)
            return len(frames) - 1

        if isinstance(exog, dict):
            for j, name in enumerate(names):
                if name not in exog or (exog[name] is not None and name not in e_train):
                    exog_state[j] = 2
                elif exog[name] is not None:
                    v = exog[name]
                    if id(v) not in obj_of_id:
                        obj_of_id[id(v)] = add_object(v)
                    exog_obj_of_col[j] = obj_of_id[id(v)]
                    exog_state[j] = 1
        else:
            assert exog.index[:n_span].equals(span_index[:len(exog)])
            exog_obj_of_col[:] = add_object(exog)
            exog_state[:] = 1
        n_obj = len(frames)
        table = pd.concat(frames) if n_obj > 1 else frames[0]
        table = transform_dataframe(
            df=table, transformer=forecaster.transformer_exog, fit=False,
            inverse_transform=False,
        )
        cat = forecaster.categorical_features_names_in_
        if forecaster.categorical_features is not None and cat:
            table = table.copy(deep=False)
            table[cat] = forecaster.categorical_encoder.transform(table[cat])
        E = table.to_numpy(dtype=float).reshape(n_obj, e1 - e0, -1)
        # Extra block of NaN for the levels without exog; (time, object, exog)
        # so that the values of a step are contiguous.
        E = np.concatenate([E, np.full((1,) + E.shape[1:], np.nan)], axis=0)
        E = np.ascontiguousarray(E.transpose(1, 0, 2))
        exog_obj_of_col[exog_obj_of_col == -1] = n_obj
        raw_isna = np.array(raw_isna + [np.zeros(n_span, dtype=bool)])
        raw_rows = np.array(raw_rows + [np.zeros(n_span, dtype=bool)])
        exog_is_dict = isinstance(exog, dict)

    calendar_all = None
    if forecaster.calendar_features is not None:
        calendar_all = forecaster.calendar_features.transform(span_index).to_numpy()
    t3 = tic()

    # ------------------------------------------------------------------
    # Folds: numpy only.
    # ------------------------------------------------------------------
    n_folds = len(folds)
    max_steps = max(fold[3][1] - fold[3][0] for fold in folds)
    P = np.full((n_folds, max_steps, n_series), np.nan)
    cols_folds, levels_predict_folds = [], []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="X does not have valid feature names", category=UserWarning
        )
        for i_fold, fold in enumerate(folds):
            lw_start, lw_end = fold[2]
            test_start, test_end = fold[3]
            steps = test_end - test_start
            ok = in_train[level_cols] & covered[lw_start:lw_end][:, level_cols].all(axis=0)
            if dropna_last_window:
                ok &= ~np.isnan(Y[lw_start:lw_end][:, level_cols]).any(axis=0)
            cols = level_cols[ok]
            cols_folds.append(cols)
            levels_predict = [names[c] for c in cols]
            levels_predict_folds.append(levels_predict)
            if cols.size == 0:
                continue

            if emit_fold_warnings and not suppress_warnings:
                # The warnings of `check_predict_input`, from the masks.
                if not dropna_last_window and np.isnan(
                    Y[lw_start:lw_end][:, cols]
                ).any():
                    warnings.warn(
                        "`last_window` has missing values. Most of machine learning "
                        "models do not allow missing values. Prediction method may "
                        "either raise an error or return NaN predictions.",
                        MissingValuesWarning,
                    )
                if E is not None:
                    _warn_exog_fold(
                        names, cols, exog, exog_state,
                        exog_obj_of_col, raw_isna, raw_rows, n_obj, name_to_col,
                        span_index, test_start, test_end,
                    )

            last_window = Yt[lw_start:lw_end][:, cols]
            if order is not None:
                last_values = []
                diffed = last_window
                for _ in range(order):
                    last_values.append(diffed[-1])
                    diffed = np.diff(diffed, n=1, axis=0)
                last_window = np.concatenate(
                    [np.full((order, cols.size), np.nan), diffed], axis=0
                )
            exog_values_dict = None
            if E is not None:
                objs = exog_obj_of_col[cols].copy()
                if exog_is_dict:
                    # Objects without rows in the test set are not used.
                    empty = ~raw_rows[objs, test_start:test_end].any(axis=1)
                    objs[empty] = n_obj
                exog_values_dict = {
                    s + 1: E[test_start - e0 + s][objs] for s in range(steps)
                }
            pred = forecaster._recursive_predict(
                steps=steps,
                levels=levels_predict,
                last_window=_Window(last_window),
                exog_values_dict=exog_values_dict,
                calendar_values=(
                    None if calendar_all is None else calendar_all[test_start:test_end]
                ),
                prediction_index=span_index[test_start:test_end],
            )
            if order is not None:
                for k in range(order):
                    pred = np.cumsum(pred, axis=0, dtype=float) + last_values[-(k + 1)]
            P[i_fold, :steps, cols] = pred.T
    t4 = tic()

    if not any(levels_predict_folds):
        raise ValueError("No fold of the backtesting has a level to predict.")

    # ------------------------------------------------------------------
    # Output, once.
    # ------------------------------------------------------------------
    if has_transformer:
        # Inverse transformation: one call per level for all its folds.
        for j, name in enumerate(names):
            transformer = forecaster.transformer_series_.get(name)
            if transformer is not None and in_train[j]:
                P[:, :, j] = transform_numpy(
                    array=P[:, :, j].ravel(), transformer=transformer, fit=False,
                    inverse_transform=True,
                ).reshape(n_folds, max_steps)

    pos_parts, col_parts, fold_parts, step_parts, pred_parts = [], [], [], [], []
    for i_fold, (fold, cols) in enumerate(zip(folds, cols_folds)):
        if cols.size == 0:
            continue
        test_start, test_end = fold[3]
        n_steps = test_end - test_start - gap
        pos = np.repeat(np.arange(test_start + gap, test_end), cols.size)
        pos_parts.append(pos)
        col_parts.append(np.tile(cols, n_steps))
        fold_parts.append(np.full(pos.size, fold[0]))
        step_parts.append(pos - test_start + 1)
        pred_parts.append(P[i_fold, gap:test_end - test_start][:, cols].ravel())
    pos = np.concatenate(pos_parts)
    col = np.concatenate(col_parts)
    pred = np.concatenate(pred_parts)
    y_true = Y[pos, col]
    pred[np.isnan(y_true)] = np.nan

    _warn_levels_not_predicted_multiseries(
        forecaster=forecaster, series=series, folds=folds, span_index=span_index,
        levels=levels, levels_predict_folds=levels_predict_folds,
        levels_train_folds=[levels_train] * n_folds,
    )

    names_index = pd.Index(names)
    index = pd.MultiIndex(
        levels=[span_index._with_freq(None) if hasattr(span_index, "_with_freq") else span_index, names_index],
        codes=[pos, col], names=["idx", "level"], verify_integrity=False,
    )
    backtest_predictions = pd.DataFrame(
        {
            "fold": np.concatenate(fold_parts),
            "step": np.concatenate(step_parts),
            "y_true": y_true,
            "pred": pred,
        },
        index=index,
    )
    t5 = tic()

    for_metrics = backtest_predictions
    if cv.overlapping_folds:
        for_metrics = for_metrics.loc[~for_metrics.index.duplicated(keep="last")]
    metrics_levels = _calculate_metrics_backtesting_multiseries(
        series=series,
        predictions=for_metrics[["fold", "y_true", "pred"]].set_axis(
            ["fold", "y_true", "y_pred"], axis=1
        ),
        folds=folds, span_index=span_index, metrics=metrics, levels=levels,
        add_aggregated_metric=add_aggregated_metric,
    )
    t6 = tic()
    backtest_predictions = backtest_predictions.reset_index("level").rename_axis(
        None, axis=0
    )
    t7 = tic()
    if timings is not None:
        timings.update(
            preprocess=t_pre - t0, prologue=t1 - t_pre, fit=t2 - t1, prepare=t3 - t2, folds=t4 - t3,
            output=(t5 - t4) + (t7 - t6), metrics=t6 - t5, total=t7 - t0,
            n_folds=n_folds,
        )
    return metrics_levels, backtest_predictions
