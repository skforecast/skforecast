"""
Individual scikit-learn compatibility checks and the check registry.

Two groups of checks are defined:

- ML forecasters: exercise every skforecast machine learning forecaster
  end to end (fit / predict, and probabilistic prediction where available).
- scikit-learn API touch points: exercise the specific scikit-learn features
  skforecast relies on and that are most exposed to upstream changes
  (cloning, pipelines, column transformers, feature selection, estimator tags,
  and the hyperparameter search workflow).

The ``CHECKS`` mapping at the bottom is consumed by the entry point.
"""

from __future__ import annotations

from collections.abc import Callable

from _sklearn_compat_data import (
    make_class_series,
    make_exog,
    make_multi_series,
    make_series,
)


# ----------------------------------------------------------------------------
# Individual checks: ML forecasters
# ----------------------------------------------------------------------------
def check_recursive_forecaster() -> None:
    """ForecasterRecursive: fit, predict and probabilistic prediction."""

    from sklearn.ensemble import RandomForestRegressor

    from skforecast.preprocessing import RollingFeatures
    from skforecast.recursive import ForecasterRecursive

    y = make_series()
    exog = make_exog()
    exog_train = exog.loc[y.index]
    exog_future = exog.iloc[len(y):]
    forecaster = ForecasterRecursive(
        estimator=RandomForestRegressor(n_estimators=10, random_state=123),
        lags=7,
        window_features=RollingFeatures(stats=["mean", "std"], window_sizes=7),
    )
    forecaster.fit(y=y, exog=exog_train, store_in_sample_residuals=True)
    forecaster.predict(steps=5, exog=exog_future)
    forecaster.predict_interval(
        steps=5,
        exog=exog_future,
        interval=[0.1, 0.9],
        method="conformal",
    )


def check_direct_forecaster() -> None:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from skforecast.direct import ForecasterDirect

    y = make_series()
    forecaster = ForecasterDirect(
        estimator=Ridge(random_state=123),
        lags=7,
        steps=5,
        transformer_y=StandardScaler(),
    )
    forecaster.fit(y=y)
    forecaster.predict(steps=5)


def check_multiseries_forecaster() -> None:
    """ForecasterRecursiveMultiSeries: global model over several series."""

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    from skforecast.recursive import ForecasterRecursiveMultiSeries

    series = make_multi_series()
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(),
        lags=7,
        encoding="ordinal",
        transformer_series=StandardScaler(),
    )
    forecaster.fit(series=series)
    forecaster.predict(steps=5)
    forecaster.predict(steps=5, levels=["series_1", "series_2"])


def check_direct_multivariate_forecaster() -> None:
    """ForecasterDirectMultiVariate: several series used as predictors."""

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from skforecast.direct import ForecasterDirectMultiVariate

    series = make_multi_series()
    forecaster = ForecasterDirectMultiVariate(
        estimator=Ridge(random_state=123),
        level="series_1",
        steps=5,
        lags=7,
        transformer_series=StandardScaler(),
    )
    forecaster.fit(series=series)
    forecaster.predict(steps=5)


def check_recursive_classifier_forecaster() -> None:
    """ForecasterRecursiveClassifier: classification-based forecasting."""

    from sklearn.linear_model import LogisticRegression

    from skforecast.preprocessing import RollingFeaturesClassification
    from skforecast.recursive import ForecasterRecursiveClassifier

    y = make_class_series()
    forecaster = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(max_iter=1000),
        lags=7,
        window_features=RollingFeaturesClassification(
            stats=["proportion"], window_sizes=7
        ),
    )
    forecaster.fit(y=y)
    forecaster.predict(steps=5)
    forecaster.predict_proba(steps=5)


def check_categorical_features() -> None:
    """Built-in categorical handling relies on sklearn's OrdinalEncoder."""

    from sklearn.ensemble import HistGradientBoostingRegressor

    from skforecast.recursive import ForecasterRecursive

    y = make_series()
    exog = make_exog()
    exog_train = exog.loc[y.index]
    exog_future = exog.iloc[len(y):]
    forecaster = ForecasterRecursive(
        estimator=HistGradientBoostingRegressor(random_state=123),
        lags=7,
        categorical_features="auto",
    )
    forecaster.fit(y=y, exog=exog_train)
    forecaster.predict(steps=5, exog=exog_future)


# ----------------------------------------------------------------------------
# Individual checks: scikit-learn API touch points and workflows
# ----------------------------------------------------------------------------
def check_pipeline_and_column_transformer() -> None:
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from skforecast.recursive import ForecasterRecursive

    y = make_series()
    exog = make_exog()
    exog_train = exog.loc[y.index]
    exog_future = exog.iloc[len(y):]

    transformer_exog = make_column_transformer(
        (StandardScaler(), ["exog_num"]),
        (OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["exog_cat"]),
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    forecaster = ForecasterRecursive(
        estimator=make_pipeline(StandardScaler(), LinearRegression()),
        lags=7,
        transformer_exog=transformer_exog,
    )
    forecaster.fit(y=y, exog=exog_train)
    forecaster.predict(steps=5, exog=exog_future)


def check_clone_and_get_set_params() -> None:
    from sklearn.base import clone
    from sklearn.linear_model import Ridge

    from skforecast.recursive import ForecasterRecursive

    forecaster = ForecasterRecursive(estimator=Ridge(), lags=7)
    cloned = clone(forecaster.estimator)
    cloned.get_params()
    cloned.set_params(alpha=0.5)


def check_backtesting() -> None:
    from sklearn.linear_model import LinearRegression

    from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
    from skforecast.recursive import ForecasterRecursive

    y = make_series()
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=7)
    cv = TimeSeriesFold(steps=5, initial_train_size=len(y) - 20, refit=False)
    backtesting_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        metric="mean_absolute_error",
        verbose=False,
        show_progress=False,
    )


def check_grid_search() -> None:
    from sklearn.ensemble import RandomForestRegressor

    from skforecast.model_selection import TimeSeriesFold, grid_search_forecaster
    from skforecast.recursive import ForecasterRecursive

    y = make_series()
    forecaster = ForecasterRecursive(
        estimator=RandomForestRegressor(random_state=123),
        lags=7,
    )
    cv = TimeSeriesFold(steps=5, initial_train_size=len(y) - 20, refit=False)
    grid_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        param_grid={"n_estimators": [5, 10], "max_depth": [3, 5]},
        lags_grid=[3, 7],
        metric="mean_absolute_error",
        return_best=True,
        verbose=False,
        show_progress=False,
    )


def check_feature_selection() -> None:
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LinearRegression

    from skforecast.feature_selection import select_features
    from skforecast.recursive import ForecasterRecursive

    y = make_series()
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=7)
    select_features(
        forecaster=forecaster,
        selector=RFECV(estimator=LinearRegression(), step=1, cv=2),
        y=y,
        verbose=False,
    )


def check_estimator_tags() -> None:
    """Estimator tags API changed across recent scikit-learn releases."""

    from sklearn.linear_model import LinearRegression

    estimator = LinearRegression()
    # ``__sklearn_tags__`` (1.6+) or the legacy ``_get_tags`` fallback.
    if hasattr(estimator, "__sklearn_tags__"):
        estimator.__sklearn_tags__()
    elif hasattr(estimator, "_get_tags"):
        estimator._get_tags()
    else:  # pragma: no cover - defensive
        raise AttributeError(
            "No estimator tags API found on scikit-learn estimator."
        )


# Registry consumed by the entry point. Order is preserved in the report.
CHECKS: dict[str, Callable[[], None]] = {
    # ML forecasters
    "ForecasterRecursive fit/predict/interval": check_recursive_forecaster,
    "ForecasterDirect + transformer_y": check_direct_forecaster,
    "ForecasterRecursiveMultiSeries": check_multiseries_forecaster,
    "ForecasterDirectMultiVariate": check_direct_multivariate_forecaster,
    "ForecasterRecursiveClassifier": check_recursive_classifier_forecaster,
    "categorical_features (OrdinalEncoder)": check_categorical_features,
    # scikit-learn API touch points and workflows
    "Pipeline + ColumnTransformer exog": check_pipeline_and_column_transformer,
    "clone / get_params / set_params": check_clone_and_get_set_params,
    "backtesting_forecaster": check_backtesting,
    "grid_search_forecaster": check_grid_search,
    "select_features (RFECV)": check_feature_selection,
    "estimator tags API": check_estimator_tags,
}
