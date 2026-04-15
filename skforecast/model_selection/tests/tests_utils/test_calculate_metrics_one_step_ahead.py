# Unit test _calculate_metrics_one_step_ahead
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
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
    transform for both differentiation and transformer_y, and y_train
    inverse transform for MASE metric (needs_y_train=True).
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
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )
    results = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
    )
    results = np.array([float(result) for result in results])

    expected = np.array([0.5516310508466604, 1.2750659053445799, 2.811352223272513])

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
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )
    results = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
    )
    results = np.array([float(result) for result in results])
    expected = np.array([0.3277718194807295, 1.3574261666383498, 0.767982227299475])

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
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )
    results = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
    )

    expected = np.array([0.5030743819038218])

    results = np.array([float(r) for r in results])
    np.testing.assert_array_almost_equal(results, expected)


def test_calculate_metrics_one_step_ahead_when_metric_does_not_need_y_train():
    """
    Test _calculate_metrics_one_step_ahead when metric does not need y_train
    (needs_y_train=False). Verifies that y_train inverse transforms are
    skipped and the metric receives y_train=None.
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
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )
    results = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
    )

    expected = np.array([1.5427323873570953])

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
        sample_weight_w, fit_kwargs_w
    ) = forecaster_weighted._train_test_split_one_step_ahead(
        y=y, initial_train_size=20
    )
    (
        X_train_u, y_train_u, X_test_u, y_test_u,
        sample_weight_u, fit_kwargs_u
    ) = forecaster_unweighted._train_test_split_one_step_ahead(
        y=y, initial_train_size=20
    )

    results_weighted = _calculate_metrics_one_step_ahead(
        forecaster=forecaster_weighted, metrics=metrics,
        X_train=X_train_w, y_train=y_train_w,
        X_test=X_test_w, y_test=y_test_w,
        sample_weight=sample_weight_w, fit_kwargs=fit_kwargs_w,
    )
    results_unweighted = _calculate_metrics_one_step_ahead(
        forecaster=forecaster_unweighted, metrics=metrics,
        X_train=X_train_u, y_train=y_train_u,
        X_test=X_test_u, y_test=y_test_u,
        sample_weight=sample_weight_u, fit_kwargs=fit_kwargs_u,
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
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog_cat, initial_train_size=20
    )

    assert 'categorical_feature' in fit_kwargs

    results = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
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
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog_cat, initial_train_size=20
    )

    assert 'categorical_feature' in fit_kwargs

    results = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
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
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y_clf, exog=exog_cat, initial_train_size=20
    )

    assert 'categorical_feature' in fit_kwargs

    results = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
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
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog_cat, initial_train_size=20
    )

    assert 'cat_features' in fit_kwargs
    assert X_train.dtype == object

    results = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
    )

    expected = np.array([0.19840676971404364])

    results = np.array([float(r) for r in results])
    np.testing.assert_array_almost_equal(results, expected)
