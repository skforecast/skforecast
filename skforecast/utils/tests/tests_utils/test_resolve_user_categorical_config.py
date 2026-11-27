# Unit test _resolve_user_categorical_config
# ==============================================================================
import re
import pytest
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from skforecast.utils import _resolve_user_categorical_config

feature_names = ['lag_1', 'lag_2', '_level_skforecast', 'exog_1', 'exog_cat']


@pytest.mark.parametrize(
    "estimator",
    [LinearRegression(),
     make_pipeline(StandardScaler(), LGBMRegressor(verbose=-1)),
     make_pipeline(
         StandardScaler(), HistGradientBoostingRegressor(categorical_features=[4])
     )],
    ids=['LinearRegression', 'Pipeline LGBMRegressor',
         'Pipeline HistGradientBoostingRegressor']
)
def test_resolve_user_categorical_config_nothing_when_not_supported_or_Pipeline(
    estimator
):
    """
    Test that nothing is configured when the estimator has no native
    categorical support or it is a Pipeline, also when the series identifier
    has to be added.
    """
    fit_kwargs = {}
    results = _resolve_user_categorical_config(
                  estimator     = estimator,
                  fit_kwargs    = fit_kwargs,
                  feature_names = feature_names,
                  level_name    = '_level_skforecast'
              )

    assert results == ({}, {}, [])
    assert results[0] is fit_kwargs


@pytest.mark.parametrize(
    "fit_kwargs, level_name, expected_fit_kwargs, expected_cat_indices",
    [({}, None, {}, []),
     ({'categorical_feature': 'auto'}, None, {'categorical_feature': 'auto'}, []),
     ({'categorical_feature': [4]}, None, {'categorical_feature': [4]}, [4]),
     ({'categorical_feature': ['exog_cat']}, None, {'categorical_feature': [4]}, [4]),
     ({'categorical_feature': ['exog_cat', 3]}, None,
      {'categorical_feature': [4, 3]}, [3, 4]),
     ({}, '_level_skforecast', {'categorical_feature': [2]}, [2]),
     ({'categorical_feature': 'auto'}, '_level_skforecast',
      {'categorical_feature': [2]}, [2]),
     ({'categorical_feature': ['exog_cat']}, '_level_skforecast',
      {'categorical_feature': [4, 2]}, [2, 4]),
     ({'categorical_feature': [2, 4]}, '_level_skforecast',
      {'categorical_feature': [2, 4]}, [2, 4])],
    ids=lambda value: f'{value}'
)
def test_resolve_user_categorical_config_output_when_LGBMRegressor(
    fit_kwargs, level_name, expected_fit_kwargs, expected_cat_indices
):
    """
    Test that, with LightGBM, the names in `categorical_feature` of
    `fit_kwargs` are translated to indices and the series identifier is added
    to them without removing those of the user.
    """
    estimator = LGBMRegressor(verbose=-1)
    results = _resolve_user_categorical_config(
                  estimator     = estimator,
                  fit_kwargs    = fit_kwargs,
                  feature_names = feature_names,
                  level_name    = level_name
              )

    assert results == (expected_fit_kwargs, {}, expected_cat_indices)


@pytest.mark.parametrize(
    "params, fit_kwargs, level_name, expected_fit_kwargs, expected_cat_indices",
    [({'categorical_feature': [4]}, {}, None, {}, [4]),
     ({'categorical_feature': '3,4'}, {}, None, {}, [3, 4]),
     ({'categorical_column': [4]}, {}, None, {}, [4]),
     ({'cat_feature': [4]}, {'categorical_feature': [3]}, None,
      {'categorical_feature': [3]}, [3]),
     ({'categorical_feature': ['exog_cat']}, {}, None,
      {'categorical_feature': [4]}, [4]),
     ({'categorical_feature': 'name:exog_1,exog_cat'}, {}, None,
      {'categorical_feature': [3, 4]}, [3, 4]),
     ({'categorical_feature': [4]}, {}, '_level_skforecast',
      {'categorical_feature': [4, 2]}, [2, 4]),
     ({'cat_column': ['exog_cat']}, {}, '_level_skforecast',
      {'categorical_feature': [4, 2]}, [2, 4]),
     ({'categorical_feature': 'auto'}, {}, '_level_skforecast',
      {'categorical_feature': [2]}, [2])],
    ids=lambda value: f'{value}'
)
def test_resolve_user_categorical_config_output_when_LGBMRegressor_constructor(
    params, fit_kwargs, level_name, expected_fit_kwargs, expected_cat_indices
):
    """
    Test that the categorical features set in the constructor of LightGBM, with
    any of the aliases of the param, are used when `fit_kwargs` has none. If
    they are positions and nothing has to be added, `fit_kwargs` is not
    changed. If they are names, or the series identifier has to be added to
    them, the positions are passed in `fit_kwargs`, which overrides the value
    of the constructor. The estimator is not modified.
    """
    estimator = LGBMRegressor(verbose=-1, **params)
    results = _resolve_user_categorical_config(
                  estimator     = estimator,
                  fit_kwargs    = fit_kwargs,
                  feature_names = feature_names,
                  level_name    = level_name
              )

    assert results == (expected_fit_kwargs, {}, expected_cat_indices)
    for name, value in params.items():
        assert estimator.get_params()[name] == value


@pytest.mark.parametrize(
    "fit_kwargs, level_name, expected_fit_kwargs, expected_cat_indices",
    [({}, None, {}, []),
     ({'cat_features': [4]}, None, {'cat_features': [4]}, [4]),
     ({'cat_features': ['exog_cat']}, None, {'cat_features': [4]}, [4]),
     ({}, '_level_skforecast', {'cat_features': [2]}, [2]),
     ({'cat_features': ['exog_cat']}, '_level_skforecast',
      {'cat_features': [4, 2]}, [2, 4])],
    ids=lambda value: f'{value}'
)
def test_resolve_user_categorical_config_output_when_CatBoostRegressor(
    fit_kwargs, level_name, expected_fit_kwargs, expected_cat_indices
):
    """
    Test that, with CatBoost, the names in `cat_features` of `fit_kwargs` are
    translated to indices and the series identifier is added to them.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    results = _resolve_user_categorical_config(
                  estimator     = estimator,
                  fit_kwargs    = fit_kwargs,
                  feature_names = feature_names,
                  level_name    = level_name
              )

    assert results == (expected_fit_kwargs, {}, expected_cat_indices)


@pytest.mark.parametrize(
    "cat_features, fit_kwargs, level_name, expected_cat_indices",
    [((4,), {}, None, [4]),
     (('exog_cat',), {}, None, [4]),
     (('exog_cat',), {'cat_features': ['exog_cat']}, None, [4]),
     ((2, 4), {}, '_level_skforecast', [2, 4]),
     (('_level_skforecast', 'exog_cat'), {}, '_level_skforecast', [2, 4])],
    ids=lambda value: f'{value}'
)
def test_resolve_user_categorical_config_output_when_CatBoostRegressor_constructor(
    cat_features, fit_kwargs, level_name, expected_cat_indices
):
    """
    Test that, when `cat_features` is set in the constructor of CatBoost, the
    estimator and `fit_kwargs` are not changed (CatBoost raises an error if
    those of `fit` differ from those of the constructor) and their indices are
    returned.
    """
    estimator = CatBoostRegressor(
        verbose=0, allow_writing_files=False, cat_features=cat_features
    )
    expected_fit_kwargs = dict(fit_kwargs)
    results = _resolve_user_categorical_config(
                  estimator     = estimator,
                  fit_kwargs    = fit_kwargs,
                  feature_names = feature_names,
                  level_name    = level_name
              )

    assert results == (expected_fit_kwargs, {}, expected_cat_indices)
    assert estimator.get_params()['cat_features'] == cat_features


def test_resolve_user_categorical_config_ValueError_when_CatBoost_constructor_without_level():
    """
    Test ValueError is raised when the series identifier has to be added and
    `cat_features` of the constructor of CatBoost does not include it.
    """
    estimator = CatBoostRegressor(
        verbose=0, allow_writing_files=False, cat_features=(4,)
    )

    err_msg = re.escape(
        "With `encoding='ordinal_category'` and `categorical_features=None`, "
        "the series identifier (`_level_skforecast`) is added to the "
        "categorical features of the estimator, which is not possible when "
        "`cat_features` is set in the constructor of the CatBoost estimator. "
        "Pass them with `fit_kwargs={'cat_features': [...]}` instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        _resolve_user_categorical_config(
            estimator     = estimator,
            fit_kwargs    = {},
            feature_names = feature_names,
            level_name    = '_level_skforecast'
        )


@pytest.mark.parametrize(
    "feature_types, level_name, expected_params, expected_cat_indices",
    [(None, None, {}, []),
     (['q', 'q', 'q', 'q', 'c'], None, {}, [4]),
     (None, '_level_skforecast',
      {'feature_types': ['q', 'q', 'c', 'q', 'q'], 'enable_categorical': True},
      [2]),
     (['q', 'q', 'q', 'q', 'c'], '_level_skforecast',
      {'feature_types': ['q', 'q', 'c', 'q', 'c'], 'enable_categorical': True},
      [2, 4]),
     (['q', 'q', 'c', 'q', 'c'], '_level_skforecast', {}, [2, 4])],
    ids=lambda value: f'{value}'
)
def test_resolve_user_categorical_config_output_when_XGBRegressor(
    feature_types, level_name, expected_params, expected_cat_indices
):
    """
    Test that, with XGBoost, the series identifier is added to `feature_types`
    of the user. The params are returned, the estimator is not modified.
    """
    estimator = XGBRegressor(feature_types=feature_types)
    results = _resolve_user_categorical_config(
                  estimator     = estimator,
                  fit_kwargs    = {},
                  feature_names = feature_names,
                  level_name    = level_name
              )

    assert results == ({}, expected_params, expected_cat_indices)
    assert estimator.get_params()['feature_types'] == feature_types


def test_resolve_user_categorical_config_ValueError_when_XGBRegressor_feature_types_length():
    """
    Test ValueError is raised when the series identifier has to be added and
    `feature_types` of the user does not have one element per feature.
    """
    estimator = XGBRegressor(feature_types=['c', 'q', 'q'])

    err_msg = re.escape(
        "`feature_types` of the estimator has 3 elements, but the training "
        "matrix has 5 features: ['lag_1', 'lag_2', '_level_skforecast', "
        "'exog_1', 'exog_cat']. Set one element per feature."
    )
    with pytest.raises(ValueError, match=err_msg):
        _resolve_user_categorical_config(
            estimator     = estimator,
            fit_kwargs    = {},
            feature_names = feature_names,
            level_name    = '_level_skforecast'
        )


@pytest.mark.parametrize(
    "categorical_features, level_name, expected_params, expected_cat_indices",
    [('from_dtype', None, {}, []),
     (None, None, {}, []),
     ([4], None, {}, [4]),
     (np.array([False, False, False, False, True]), None, {}, [4]),
     (['exog_cat'], None, {'categorical_features': [4]}, [4]),
     ('from_dtype', '_level_skforecast', {'categorical_features': [2]}, [2]),
     ([4], '_level_skforecast', {'categorical_features': [2, 4]}, [2, 4]),
     (np.array([4]), '_level_skforecast', {'categorical_features': [2, 4]}, [2, 4]),
     (['exog_cat'], '_level_skforecast', {'categorical_features': [2, 4]}, [2, 4]),
     (np.array([False, False, False, False, True]), '_level_skforecast',
      {'categorical_features': [2, 4]}, [2, 4]),
     ([2, 4], '_level_skforecast', {}, [2, 4])],
    ids=lambda value: f'{value}'
)
def test_resolve_user_categorical_config_output_when_HistGradientBoostingRegressor(
    categorical_features, level_name, expected_params, expected_cat_indices
):
    """
    Test that, with HistGradientBoosting, the names in `categorical_features`
    of the estimator are translated to indices and the series identifier is
    added to them. The params are returned, the estimator is not modified.
    """
    estimator = HistGradientBoostingRegressor(
        categorical_features=categorical_features
    )
    results = _resolve_user_categorical_config(
                  estimator     = estimator,
                  fit_kwargs    = {},
                  feature_names = feature_names,
                  level_name    = level_name
              )

    assert results == ({}, expected_params, expected_cat_indices)
    assert estimator.get_params()['categorical_features'] is categorical_features


def test_resolve_user_categorical_config_ValueError_when_name_not_in_features():
    """
    Test ValueError is raised when a categorical feature given by name is not
    a feature of the training matrix.
    """
    err_msg = re.escape(
        "`categorical_feature` contains names that are not features of the "
        "training matrix: ['not_a_feature']. Features: ['lag_1', 'lag_2', "
        "'_level_skforecast', 'exog_1', 'exog_cat']."
    )
    with pytest.raises(ValueError, match=err_msg):
        _resolve_user_categorical_config(
            estimator     = LGBMRegressor(verbose=-1),
            fit_kwargs    = {'categorical_feature': ['not_a_feature']},
            feature_names = feature_names
        )
