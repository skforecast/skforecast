# Unit test _calculate_metrics_one_step_ahead
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterRecursiveClassifier
from skforecast.direct import ForecasterDirect
from skforecast.model_selection._utils import _calculate_metrics_one_step_ahead
from skforecast.metrics import add_y_train_argument

# Fixtures
from ..fixtures_model_selection import y
from ..fixtures_model_selection import y_clf
from ..fixtures_model_selection import exog


def test_calculate_metrics_one_step_ahead_when_ForecasterRecursive():
    """
    Test _calculate_metrics_one_step_ahead with ForecasterRecursive using
    transformer_y, transformer_exog, and differentiation. Covers inverse
    transform for both differentiation and transformer_y: each prediction is
    reverted one step ahead from the observed values, so the metrics are
    equal to a backtesting with `steps=1` and `refit=False`. MASE scales the
    error of each prediction with the naive forecast over the values of `y`
    before it (its forecast origin is its own position).
    """

    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=5,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
        differentiation=1,
    )
    metrics = [
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_absolute_scaled_error,
    ]
    metrics = [add_y_train_argument(metric) for metric in metrics]
    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs, test_index
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )
    results, _ = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
        y=y,
        test_index=test_index,
    )
    results = np.array([float(result) for result in results])

    expected = np.array([0.3216560967772606, 1.1612907172094225, 1.3128815728782959])

    np.testing.assert_array_almost_equal(results, expected)


def test_calculate_metrics_one_step_ahead_when_ForecasterDirect():
    """
    Test _calculate_metrics_one_step_ahead with ForecasterDirect (steps=3).
    Verifies estimators_[1] is used instead of estimator, and transformer_y
    inverse transform is applied (no differentiation).
    """

    forecaster = ForecasterDirect(
        estimator=LinearRegression(),
        lags=5,
        steps=3,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    metrics = [
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_absolute_scaled_error,
    ]
    metrics = [add_y_train_argument(metric) for metric in metrics]
    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs, test_index
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )
    results, _ = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
        y=y,
        test_index=test_index,
    )
    results = np.array([float(result) for result in results])
    expected = np.array([0.3277718194807295, 1.3574261666383498, 1.3342638531294446])

    np.testing.assert_array_almost_equal(results, expected)


def test_calculate_metrics_one_step_ahead_when_ForecasterRecursive_only_differentiation():
    """
    Test _calculate_metrics_one_step_ahead with ForecasterRecursive using
    differentiation but no transformer_y. Covers the differentiation
    inverse transform path without the transformer_y path.
    """

    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=5,
        differentiation=1,
    )
    metrics = [add_y_train_argument(mean_absolute_error)]
    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs, test_index
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )
    results, _ = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
        y=y,
        test_index=test_index,
    )

    expected = np.array([0.3206559378236501])

    results = np.array([float(r) for r in results])
    np.testing.assert_array_almost_equal(results, expected)


def test_calculate_metrics_one_step_ahead_when_metric_does_not_need_y_train():
    """
    Test _calculate_metrics_one_step_ahead when metric does not need y_train
    (needs_y_train=False). Verifies that the metric receives y_train=None.
    """

    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=5,
        transformer_y=StandardScaler(),
        differentiation=1,
    )
    # mean_absolute_error does not need y_train
    metrics = [add_y_train_argument(mean_absolute_error)]
    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs, test_index
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )
    results, _ = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
        y=y,
        test_index=test_index,
    )

    expected = np.array([0.351186353406055])

    results = np.array([float(r) for r in results])
    np.testing.assert_array_almost_equal(results, expected)


def test_calculate_metrics_one_step_ahead_when_weight_func_non_uniform():
    """
    Test _calculate_metrics_one_step_ahead with non-uniform sample weights.
    Verifies that passing sample_weight actually changes the fitted model
    and therefore the metric results compared to the unweighted version.
    """

    def custom_weights(index):
        weights = np.where(
            index >= index[len(index) // 2],
            10.0,
            1.0
        )
        return weights

    forecaster_weighted = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=5,
        weight_func=custom_weights,
    )
    forecaster_unweighted = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=5,
    )
    metrics = [add_y_train_argument(mean_absolute_error)]

    (
        X_train_w, y_train_w, X_test_w, y_test_w,
        sample_weight_w, fit_kwargs_w, test_index_w
    ) = forecaster_weighted._train_test_split_one_step_ahead(
        y=y, initial_train_size=20
    )
    (
        X_train_u, y_train_u, X_test_u, y_test_u,
        sample_weight_u, fit_kwargs_u, test_index_u
    ) = forecaster_unweighted._train_test_split_one_step_ahead(
        y=y, initial_train_size=20
    )

    results_weighted, _ = _calculate_metrics_one_step_ahead(
        forecaster=forecaster_weighted, metrics=metrics,
        X_train=X_train_w, y_train=y_train_w,
        X_test=X_test_w, y_test=y_test_w,
        sample_weight=sample_weight_w, fit_kwargs=fit_kwargs_w,
        y=y, test_index=test_index_w,
    )
    results_unweighted, _ = _calculate_metrics_one_step_ahead(
        forecaster=forecaster_unweighted, metrics=metrics,
        X_train=X_train_u, y_train=y_train_u,
        X_test=X_test_u, y_test=y_test_u,
        sample_weight=sample_weight_u, fit_kwargs=fit_kwargs_u,
        y=y, test_index=test_index_u,
    )

    expected_weighted = np.array([0.2736986610006486])
    expected_unweighted = np.array([0.20742401958889])

    np.testing.assert_array_almost_equal(results_weighted, expected_weighted)
    np.testing.assert_array_almost_equal(results_unweighted, expected_unweighted)


def test_calculate_metrics_one_step_ahead_when_ForecasterRecursive_with_categorical():
    """
    Test _calculate_metrics_one_step_ahead with ForecasterRecursive using
    LGBMRegressor and categorical_features. Verifies that fit_kwargs with
    categorical_feature indices are correctly passed to estimator.fit().
    """

    n = len(y)
    exog_cat = pd.DataFrame({
        'exog_num': exog.values,
        'exog_cat': pd.Categorical(([0, 1, 2] * ((n // 3) + 1))[:n]),
    }, index=y.index)

    forecaster = ForecasterRecursive(
        estimator=LGBMRegressor(n_estimators=10, random_state=123, verbose=-1),
        lags=5,
        categorical_features='auto',
    )
    metrics = [add_y_train_argument(mean_absolute_error)]
    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs, test_index
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog_cat, initial_train_size=20
    )

    assert 'categorical_feature' in fit_kwargs

    results, _ = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
        y=y,
        test_index=test_index,
    )

    expected = np.array([0.19085790731153274])

    results = np.array([float(r) for r in results])
    np.testing.assert_array_almost_equal(results, expected)


def test_calculate_metrics_one_step_ahead_when_ForecasterDirect_with_categorical():
    """
    Test _calculate_metrics_one_step_ahead with ForecasterDirect using
    LGBMRegressor and categorical_features. Verifies that fit_kwargs with
    categorical_feature indices are correctly passed to estimators_[1].fit().
    """

    n = len(y)
    exog_cat = pd.DataFrame({
        'exog_num': exog.values,
        'exog_cat': pd.Categorical(([0, 1, 2] * ((n // 3) + 1))[:n]),
    }, index=y.index)

    forecaster = ForecasterDirect(
        estimator=LGBMRegressor(n_estimators=10, random_state=123, verbose=-1),
        lags=5,
        steps=3,
        categorical_features='auto',
    )
    metrics = [add_y_train_argument(mean_absolute_error)]
    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs, test_index
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog_cat, initial_train_size=20
    )

    assert 'categorical_feature' in fit_kwargs

    results, _ = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
        y=y,
        test_index=test_index,
    )

    expected = np.array([0.1978084357142857])

    results = np.array([float(r) for r in results])
    np.testing.assert_array_almost_equal(results, expected)


def test_calculate_metrics_one_step_ahead_when_ForecasterRecursiveClassifier():
    """
    Test _calculate_metrics_one_step_ahead with ForecasterRecursiveClassifier.
    Verifies the function works with classification metrics and classifiers,
    using LGBMClassifier with native categorical support for lags.
    """

    n = len(y_clf)
    exog_cat = pd.DataFrame({
        'exog_num': exog.values,
        'exog_cat': pd.Categorical(([0, 1, 2] * ((n // 3) + 1))[:n]),
    }, index=y_clf.index)

    forecaster = ForecasterRecursiveClassifier(
        estimator=LGBMClassifier(n_estimators=10, random_state=123, verbose=-1),
        lags=5,
        features_encoding='auto',
        categorical_features='auto',
    )
    metrics = [add_y_train_argument(mean_absolute_error)]
    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs, test_index
    ) = forecaster._train_test_split_one_step_ahead(
        y=y_clf, exog=exog_cat, initial_train_size=20
    )

    assert 'categorical_feature' in fit_kwargs

    results, _ = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
        y=y_clf,
        test_index=test_index,
    )

    expected = np.array([1.0333333333333334])

    results = np.array([float(r) for r in results])
    np.testing.assert_array_almost_equal(results, expected)


def test_calculate_metrics_one_step_ahead_when_catboost_categorical():
    """
    Test _calculate_metrics_one_step_ahead with CatBoostRegressor and
    categorical_features. Verifies that fit_kwargs with cat_features and
    int-casted X_train columns work correctly through the full pipeline.
    """

    n = len(y)
    exog_cat = pd.DataFrame({
        'exog_num': exog.values,
        'exog_cat': pd.Categorical(([0, 1, 2] * ((n // 3) + 1))[:n]),
    }, index=y.index)

    forecaster = ForecasterRecursive(
        estimator=CatBoostRegressor(
            iterations=10, random_state=123, verbose=0, allow_writing_files=False
        ),
        lags=5,
        categorical_features='auto',
    )
    metrics = [add_y_train_argument(mean_absolute_error)]
    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs, test_index
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog_cat, initial_train_size=20
    )

    assert 'cat_features' in fit_kwargs
    assert X_train.dtype == object

    results, _ = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
        y=y,
        test_index=test_index,
    )

    expected = np.array([0.19840676971404364])

    results = np.array([float(r) for r in results])
    np.testing.assert_array_almost_equal(results, expected)


def test_calculate_metrics_one_step_ahead_n_evaluated_when_y_has_NaN_and_dropna_from_series():
    """
    Test the number of points evaluated returned by
    _calculate_metrics_one_step_ahead when `y` has missing values and
    `dropna_from_series=True`: the rows of the test set whose target or lags
    are missing are removed, so a larger window evaluates fewer points. With
    NaN at positions 25 and 35 and a test set from position 20 (30 rows),
    lags=3 removes positions 25 to 28 and 35 to 38 (22 points left) and
    lags=10 removes positions 25 to 45 (9 points left).
    """
    y_nan = y.copy()
    y_nan.iloc[[25, 35]] = np.nan
    metrics = [add_y_train_argument(mean_absolute_error)]

    n_evaluated = []
    for lags in [3, 10]:
        forecaster = ForecasterRecursive(
            estimator=LinearRegression(), lags=lags, dropna_from_series=True
        )
        (
            X_train, y_train, X_test, y_test, sample_weight, fit_kwargs, test_index
        ) = forecaster._train_test_split_one_step_ahead(
            y=y_nan, initial_train_size=20
        )
        _, n = _calculate_metrics_one_step_ahead(
            forecaster=forecaster,
            metrics=metrics,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sample_weight=sample_weight,
            fit_kwargs=fit_kwargs,
            y=y_nan,
            test_index=test_index,
        )
        n_evaluated.append(n)

    assert n_evaluated == [22, 9]


def test_calculate_metrics_one_step_ahead_scaled_metrics_scaled_by_own_position():
    """
    Test that `mean_absolute_scaled_error` scales the error of each one-step
    prediction with the naive forecast over all the values of `y` before it
    (its forecast origin is its own position), for ForecasterRecursive and
    ForecasterDirect (whose test set loses its last `steps - 1` rows): the
    scale does not depend on `window_size` or `steps`.
    """
    metrics = [add_y_train_argument(mean_absolute_scaled_error)]
    values = y.to_numpy()
    for forecaster in [
        ForecasterRecursive(estimator=LinearRegression(), lags=10),
        ForecasterDirect(estimator=LinearRegression(), lags=10, steps=5),
    ]:
        (
            X_train, y_train, X_test, y_test, sample_weight, fit_kwargs, test_index
        ) = forecaster._train_test_split_one_step_ahead(y=y, initial_train_size=20)
        results, _ = _calculate_metrics_one_step_ahead(
            forecaster=forecaster,
            metrics=metrics,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sample_weight=sample_weight,
            fit_kwargs=fit_kwargs,
            y=y,
            test_index=test_index,
        )

        estimator = (
            forecaster.estimators_[1]
            if isinstance(forecaster, ForecasterDirect)
            else forecaster.estimator
        )
        errors = y.loc[test_index].to_numpy() - estimator.predict(X_test)
        positions = y.index.get_indexer(test_index)
        scale = np.array([np.mean(np.abs(np.diff(values[:p]))) for p in positions])
        np.testing.assert_almost_equal(results[0], np.mean(np.abs(errors) / scale))


def test_calculate_metrics_one_step_ahead_differentiation_missing_values_in_test():
    """
    Test that, with differentiation and missing values in the test set, the
    true values are the observed ones and each prediction is the previous
    observed value plus the predicted difference. The rows with a missing
    target are left out of the test set, so the true values cannot be rebuilt
    with the cumulative sum of the differences. (A backtesting cannot be the
    reference: it raises an error when the last value before a date is
    missing, since the differentiation cannot be reverted.)
    """
    rng = np.random.default_rng(123)
    y_nan = pd.Series(
        50 + rng.normal(size=80).cumsum(),
        index=pd.date_range("2020-01-01", periods=80, freq="D"),
        name="y",
    )
    y_nan.iloc[[50, 51, 65]] = np.nan
    forecaster = ForecasterRecursive(
        estimator=HistGradientBoostingRegressor(max_iter=20, random_state=123),
        lags=5,
        differentiation=1,
    )
    received = {}

    def capture(y_true, y_pred):
        received["y_true"] = np.asarray(y_true)
        received["y_pred"] = np.asarray(y_pred)
        return 0.0

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs, test_index
    ) = forecaster._train_test_split_one_step_ahead(
        y=y_nan, exog=None, initial_train_size=40
    )
    _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=[add_y_train_argument(capture)],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
        y=y_nan,
        test_index=test_index,
    )

    diff_pred = forecaster.estimator.predict(X_test)
    expected_y_pred = y_nan.shift(1).loc[test_index].to_numpy() + diff_pred

    assert len(test_index) < len(y_nan) - 40
    np.testing.assert_array_almost_equal(
        received["y_true"], y_nan.loc[test_index].to_numpy()
    )
    np.testing.assert_array_almost_equal(received["y_pred"], expected_y_pred)
