"""
Second prototype of the multi-series backtesting that validates and prepares
once. It extends `proto_bt_once.py` with:

- **refit** (`refit=True` or an integer): the series matrix, the masks and the
  alignment of exog are prepared once (`prepare_static`); what depends on the
  fitted model (transformed series, exog encoded for the test rows of the
  block of folds it predicts) is prepared after each fit.
- **a preparation that can be shared** by several backtestings over the same
  data and folds (the candidates of a hyperparameter search): `static=`.
- **metrics in numpy** (`fast_metrics=True`), only for `mean_absolute_error`
  and `mean_squared_error`, to measure what is left once BTPLAN M1 and PERF-03
  are done. It is compared with the current function in the callers.

Scope: `initial_train_size` not None, `interval=None`,
`return_predictors=False`, `n_jobs=1`, `ForecasterRecursiveMultiSeries`, a
single differentiation order. Nothing in `skforecast/` is modified.
"""
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from proto_bt_once import _warn_exog_fold, _Window
from skforecast.exceptions import LongTrainingWarning, MissingValuesWarning
from skforecast.metrics import _get_metric, add_y_train_argument
from skforecast.model_selection._utils import (
    _calculate_metrics_backtesting_multiseries,
    _initialize_levels_model_selection_multiseries,
    _select_fit_folds_multiseries,
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

tic = time.perf_counter


def prepare_static(series, exog, cv):
    """
    Everything that does not depend on the forecaster: series and exog as
    dicts (as `backtesting_forecaster_multiseries` does), matrix of series,
    mask of the dates present in each index, and exog objects aligned with the
    span of the test sets (not encoded). `cv` must already have `window_size`.
    """
    t0 = tic()
    series, series_indexes = check_preprocess_series(series)
    names = list(series.keys())
    if exog is not None:
        exog, _ = check_preprocess_exog_multiseries(
            series_names_in_=names,
            series_index_type=type(series_indexes[names[0]]),
            exog=exog,
            exog_dict={serie: None for serie in names},
        )
    t1 = tic()
    span_index = cv._extract_index(X=series)
    n_span, n_series = len(span_index), len(names)
    Y = np.full((n_span, n_series), np.nan)
    covered = np.zeros((n_span, n_series), dtype=bool)
    for j, name in enumerate(names):
        v = series[name]
        if isinstance(v, pd.DataFrame):
            v = v.iloc[:, 0]
        pos = span_index.get_indexer(v.index)
        Y[pos, j] = v.to_numpy(dtype=float)
        covered[pos, j] = True
    st = dict(
        series=series, exog=exog, names=names, span_index=span_index, Y=Y,
        covered=covered, name_to_col={name: j for j, name in enumerate(names)},
        seconds_preprocess=t1 - t0, exog_ready=False,
    )
    st["seconds_matrix"] = tic() - t1
    return st


def _prepare_static_exog(st, e0, e1):
    """Align each distinct exog object with the rows [e0, e1) of the span."""
    t0 = tic()
    exog, names, span_index = st["exog"], st["names"], st["span_index"]
    n_span, n_series = len(span_index), len(names)
    frames, raw_isna, raw_rows, obj_of_id = [], [], [], {}
    obj_of_col = np.full(n_series, -1)
    has_key = np.zeros(n_series, dtype=bool)
    for j, name in enumerate(names):
        v = exog.get(name)
        has_key[j] = name in exog
        if v is None:
            continue
        if id(v) not in obj_of_id:
            frame = v.to_frame() if isinstance(v, pd.Series) else v
            pos = span_index.get_indexer(frame.index)
            rows = np.zeros(n_span, dtype=bool)
            rows[pos[pos >= 0]] = True
            isna = np.zeros(n_span, dtype=bool)
            isna[pos[pos >= 0]] = frame.isna().to_numpy().any(axis=1)[pos >= 0]
            frames.append(frame.reindex(index=span_index[e0:e1]))
            raw_isna.append(isna)
            raw_rows.append(rows)
            obj_of_id[id(v)] = len(frames) - 1
        obj_of_col[j] = obj_of_id[id(v)]
    n_obj = len(frames)
    st.update(
        e0=e0, e1=e1, n_obj=n_obj, obj_of_col=obj_of_col, has_key=has_key,
        table_raw=pd.concat(frames) if n_obj > 1 else frames[0],
        raw_isna=np.array(raw_isna + [np.zeros(n_span, dtype=bool)]),
        raw_rows=np.array(raw_rows + [np.zeros(n_span, dtype=bool)]),
        exog_ready=True,
    )
    st["seconds_exog"] = tic() - t0


def _extract_train(st, fold, window_size):
    """Training set of a fold: the code of `_extract_data_folds_multiseries`."""
    series, exog, span_index = st["series"], st["exog"], st["span_index"]
    start, end = span_index[fold[1][0]], span_index[fold[1][1] - 1]
    series_train = {}
    for k, v in series.items():
        v_train = v.loc[start:end]
        if not v_train.isna().to_numpy().all():
            first, last = v_train.first_valid_index(), v_train.last_valid_index()
            if first is not None and last is not None:
                v_train = v_train.loc[first:last]
                if len(v_train) > window_size:
                    series_train[k] = v_train
    exog_train = None
    if exog is not None:
        exog_train = {}
        for k, v in exog.items():
            if v is None:
                exog_train[k] = None
            else:
                v_train = v.loc[start:end]
                if len(v_train) > 0:
                    exog_train[k] = v_train
    return series_train, exog_train


def _levels_in_train(st, fold, window_size):
    """Mask of the series of the training set of a fold, in numpy."""
    notna = ~np.isnan(st["Y"][fold[1][0]:fold[1][1]])
    n = notna.shape[0]
    first = notna.argmax(axis=0)
    last = n - 1 - notna[::-1].argmax(axis=0)
    return notna.any(axis=0) & ((last - first + 1) > window_size)


def _fast_metrics(metric_names, levels, names, level_cols, col, y_true, pred, add_aggregated_metric):
    """MAE and MSE by level with numpy slices, in the format of the current function."""
    ok = ~(np.isnan(y_true) | np.isnan(pred))
    col_ok, err = col[ok], (y_true[ok] - pred[ok])
    order = np.argsort(col_ok, kind="stable")
    err = err[order]
    counts = np.bincount(col_ok, minlength=len(names))
    bounds = np.concatenate([[0], np.cumsum(counts)])
    formulas = {
        "mean_absolute_error": lambda e: np.mean(np.abs(e)),
        "mean_squared_error": lambda e: np.mean(e * e),
    }
    rows = []
    for c in level_cols:
        e = err[bounds[c]:bounds[c + 1]]
        rows.append([formulas[m](e) if e.size else None for m in metric_names])
    out = pd.DataFrame(rows, columns=metric_names)
    out.insert(0, "levels", levels)
    out["n_evaluated"] = counts[level_cols]
    if add_aggregated_metric and len(levels) > 1:
        n_eval = int(out["n_evaluated"].sum())
        with_points = out.loc[out["n_evaluated"] > 0]
        in_levels = np.isin(col_ok[order], level_cols)
        agg = {
            "average": [out[m].mean(skipna=True) for m in metric_names],
            "weighted_average": [
                np.average(with_points[m], weights=with_points["n_evaluated"])
                for m in metric_names
            ],
            "pooling": [formulas[m](err[in_levels]) for m in metric_names],
        }
        extra = pd.DataFrame(agg.values(), columns=metric_names)
        extra["levels"] = list(agg)
        extra["n_evaluated"] = n_eval
        out = pd.concat([out, extra], axis=0, ignore_index=True)
    return out


def backtesting_blocks(
    forecaster, series, cv, metric, levels=None, exog=None,
    add_aggregated_metric=True, suppress_warnings=False, static=None,
    fast_metrics=False, timings=None,
):
    """
    Same result as `backtesting_forecaster_multiseries(..., n_jobs=1,
    show_progress=False)` within the scope of the module docstring, with or
    without refit. `static` is the result of `prepare_static` (shared by the
    candidates of a search); if None it is created here.
    """
    T = dict.fromkeys(
        ("preprocess", "matrix", "exog_align", "extract_train", "fit",
         "prepare_model", "folds", "output", "metrics"), 0.0
    )
    t_start = tic()
    forecaster = deepcopy_forecaster(forecaster, include_out_sample_residuals=False)
    cv = deepcopy(cv)
    cv.set_params({"window_size": forecaster.window_size, "return_all_indexes": False})
    refit = False if cv.refit == 0 else cv.refit
    if static is None:
        static = prepare_static(series, exog, cv)
        T["preprocess"], T["matrix"] = static["seconds_preprocess"], static["seconds_matrix"]
    st = static
    series, exog, names, span_index = st["series"], st["exog"], st["names"], st["span_index"]
    Y, covered, name_to_col = st["Y"], st["covered"], st["name_to_col"]
    n_series = len(names)

    levels = _initialize_levels_model_selection_multiseries(
        forecaster=forecaster, series=series, levels=levels
    )
    metric_list = metric if isinstance(metric, list) else [metric]
    metrics = [
        _get_metric(metric=m) if isinstance(m, str) else add_y_train_argument(m)
        for m in metric_list
    ]
    forecaster._probabilistic_mode = False
    forecaster._allow_nan_predictions = True
    folds = cv.split(X=series, as_pandas=False)
    gap, window_size = cv.gap, forecaster.window_size
    dropna_last_window = (
        forecaster.differentiation_max is not None
        or not estimator_has_native_nan_support(forecaster.estimator)
    )
    level_cols = np.array([name_to_col[level] for level in levels])
    order = forecaster.differentiation
    assert order is None or isinstance(order, (int, np.integer))

    e0 = min(fold[3][0] for fold in folds)
    e1 = max(fold[3][1] for fold in folds)
    if exog is not None:
        if not st["exog_ready"]:
            _prepare_static_exog(st, e0, e1)
            T["exog_align"] = st["seconds_exog"]
        assert (st["e0"], st["e1"]) == (e0, e1)
        n_obj, obj_of_col = st["n_obj"], st["obj_of_col"]
        raw_isna, raw_rows = st["raw_isna"], st["raw_rows"]

    calendar_all = None
    if forecaster.calendar_features is not None:
        calendar_all = forecaster.calendar_features.transform(span_index).to_numpy()

    # Levels of the training set and of the last window of every fold, in
    # numpy, to know which folds are fitted before fitting any.
    n_folds = len(folds)
    in_train_folds, ok_folds = [], []
    # NOTE: As in `_extract_data_folds_multiseries`, the training set is
    # extracted again when its limits change, not when the fold is fitted:
    # with `skip_folds` and an intermittent refit the limits can change in a
    # fold that is not fitted.
    in_train, limits = None, None
    for fold in folds:
        if tuple(fold[1]) != limits:
            limits = tuple(fold[1])
            in_train = _levels_in_train(st, fold, window_size)
        lw_start, lw_end = fold[2]
        ok = in_train & covered[lw_start:lw_end].all(axis=0)
        if dropna_last_window:
            ok &= ~np.isnan(Y[lw_start:lw_end]).any(axis=0)
        in_train_folds.append(in_train)
        ok_folds.append(ok)
    cols_folds = [level_cols[ok[level_cols]] for ok in ok_folds]
    levels_predict_folds = [[names[c] for c in cols] for cols in cols_folds]
    levels_train_folds = [[names[j] for j in np.flatnonzero(m)] for m in in_train_folds]
    if not any(levels_predict_folds):
        raise ValueError("No fold of the backtesting has a level to predict.")
    fit_folds = _select_fit_folds_multiseries(
        folds=folds, levels_predict_folds=levels_predict_folds
    )
    if refit and sum(fit_folds) > 50:
        warnings.warn(
            f"The forecaster will be fit {sum(fit_folds)} times. This can take substantial "
            f"amounts of time. If not feasible, try with `refit = False`.\n",
            LongTrainingWarning,
        )

    max_steps = max(fold[3][1] - fold[3][0] for fold in folds)
    P = np.full((n_folds, max_steps, n_series), np.nan)
    model_of_fold = np.full(n_folds, -1)
    inverse_jobs = []  # (first fold, last fold + 1, transformers) of each model
    Yt = E = exog_state = exog_limits = None
    block_start = 0
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="X does not have valid feature names", category=UserWarning
        )
        for i_fold, fold in enumerate(folds):
            if fit_folds[i_fold]:
                t = tic()
                s_train, e_train = _extract_train(st, fold, window_size)
                assert list(s_train) == levels_train_folds[i_fold]
                T["extract_train"] += tic() - t
                t = tic()
                forecaster.fit(
                    series=s_train, exog=e_train,
                    store_last_window=[names[j] for j in np.flatnonzero(ok_folds[i_fold])],
                    store_in_sample_residuals=True, suppress_warnings=suppress_warnings,
                )
                T["fit"] += tic() - t

                # What depends on the model, for the folds it predicts.
                t = tic()
                block_end = i_fold + 1
                while block_end < n_folds and not fit_folds[block_end]:
                    block_end += 1
                block_start = i_fold
                in_train = in_train_folds[i_fold]
                transformers = [
                    forecaster.transformer_series_.get(name) if in_train[j] else None
                    for j, name in enumerate(names)
                ]
                Yt = Y
                if any(tr is not None for tr in transformers):
                    a = min(f[2][0] for f in folds[block_start:block_end])
                    b = max(f[2][1] for f in folds[block_start:block_end])
                    Yt = Y.copy()
                    for j, tr in enumerate(transformers):
                        if tr is not None:
                            Yt[a:b, j] = transform_numpy(
                                array=Y[a:b, j], transformer=tr, fit=False,
                                inverse_transform=False,
                            )
                    inverse_jobs.append((block_start, block_end, transformers))
                if exog is not None:
                    b0 = min(f[3][0] for f in folds[block_start:block_end])
                    b1 = max(f[3][1] for f in folds[block_start:block_end])
                    table = st["table_raw"]
                    if (b0, b1) != (e0, e1):
                        length = e1 - e0
                        rows = (
                            np.arange(n_obj)[:, None] * length
                            + np.arange(b0 - e0, b1 - e0)[None, :]
                        ).ravel()
                        table = table.iloc[rows]
                    table = table.reindex(columns=forecaster.exog_names_in_)
                    table = transform_dataframe(
                        df=table, transformer=forecaster.transformer_exog, fit=False,
                        inverse_transform=False,
                    )
                    cat = forecaster.categorical_features_names_in_
                    if forecaster.categorical_features is not None and cat:
                        table = table.copy(deep=False)
                        table[cat] = forecaster.categorical_encoder.transform(table[cat])
                    E = table.to_numpy(dtype=float).reshape(n_obj, b1 - b0, -1)
                    E = np.concatenate([E, np.full((1,) + E.shape[1:], np.nan)], axis=0)
                    E = np.ascontiguousarray(E.transpose(1, 0, 2))
                    E_offset = b0
                T["prepare_model"] += tic() - t

            cols = cols_folds[i_fold]
            if exog is not None and tuple(fold[1]) != exog_limits:
                # exog_state: 0 = None (NaN, no warning), 1 = usable,
                # 2 = no key or no rows in the training set (by its limits).
                exog_limits = tuple(fold[1])
                in_exog_train = raw_rows[:, fold[1][0]:fold[1][1]].any(axis=1)
                exog_state = np.where(
                    ~st["has_key"], 2,
                    np.where(obj_of_col == -1, 0,
                             np.where(in_exog_train[obj_of_col], 1, 2)),
                )
            if cols.size == 0:
                continue
            t = tic()
            model_of_fold[i_fold] = block_start
            lw_start, lw_end = fold[2]
            test_start, test_end = fold[3]
            steps = test_end - test_start
            if not suppress_warnings:
                if not dropna_last_window and np.isnan(Y[lw_start:lw_end][:, cols]).any():
                    warnings.warn(
                        "`last_window` has missing values. Most of machine learning "
                        "models do not allow missing values. Prediction method may "
                        "either raise an error or return NaN predictions.",
                        MissingValuesWarning,
                    )
                if exog is not None:
                    _warn_exog_fold(
                        names, cols, exog, exog_state,
                        np.where(obj_of_col == -1, n_obj, obj_of_col), raw_isna,
                        raw_rows, n_obj, name_to_col, span_index, test_start, test_end,
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
            if exog is not None:
                objs = np.where(exog_state[cols] == 1, obj_of_col[cols], n_obj)
                empty = ~raw_rows[objs, test_start:test_end].any(axis=1)
                objs[empty] = n_obj
                exog_values_dict = {
                    s + 1: E[test_start - E_offset + s][objs] for s in range(steps)
                }
            pred = forecaster._recursive_predict(
                steps=steps, levels=levels_predict_folds[i_fold],
                last_window=_Window(last_window), exog_values_dict=exog_values_dict,
                calendar_values=(
                    None if calendar_all is None else calendar_all[test_start:test_end]
                ),
                prediction_index=span_index[test_start:test_end],
            )
            if order is not None:
                for k in range(order):
                    pred = np.cumsum(pred, axis=0, dtype=float) + last_values[-(k + 1)]
            P[i_fold, :steps, cols] = pred.T
            T["folds"] += tic() - t

    # ------------------------------------------------------------------
    # Output, once.
    # ------------------------------------------------------------------
    t = tic()
    for a, b, transformers in inverse_jobs:
        for j, tr in enumerate(transformers):
            if tr is not None:
                P[a:b, :, j] = transform_numpy(
                    array=P[a:b, :, j].ravel(), transformer=tr, fit=False,
                    inverse_transform=True,
                ).reshape(b - a, max_steps)
    pos_parts, col_parts, fold_parts, step_parts, pred_parts = [], [], [], [], []
    for i_fold, (fold, cols) in enumerate(zip(folds, cols_folds)):
        if cols.size == 0:
            continue
        test_start, test_end = fold[3]
        pos = np.repeat(np.arange(test_start + gap, test_end), cols.size)
        pos_parts.append(pos)
        col_parts.append(np.tile(cols, test_end - test_start - gap))
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
        levels_train_folds=levels_train_folds,
    )
    index = pd.MultiIndex(
        levels=[
            span_index._with_freq(None) if hasattr(span_index, "_with_freq") else span_index,
            pd.Index(names),
        ],
        codes=[pos, col], names=["idx", "level"], verify_integrity=False,
    )
    backtest_predictions = pd.DataFrame(
        {
            "fold": np.concatenate(fold_parts), "step": np.concatenate(step_parts),
            "y_true": y_true, "pred": pred,
        },
        index=index,
    )
    T["output"] += tic() - t

    t = tic()
    if fast_metrics:
        assert not cv.overlapping_folds
        metrics_levels = _fast_metrics(
            metric_list, levels, names, level_cols, col, y_true, pred, add_aggregated_metric
        )
    else:
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
    T["metrics"] += tic() - t
    t = tic()
    backtest_predictions = backtest_predictions.reset_index("level").rename_axis(None, axis=0)
    T["output"] += tic() - t
    if timings is not None:
        T["total"] = tic() - t_start
        T["rest"] = T["total"] - sum(v for k, v in T.items() if k != "total")
        T["n_folds"], T["n_fits"] = n_folds, sum(fit_folds)
        timings.update(T)
    return metrics_levels, backtest_predictions
