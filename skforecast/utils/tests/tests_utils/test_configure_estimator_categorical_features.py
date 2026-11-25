# Unit test configure_estimator_categorical_features
# ==============================================================================
import re
import pytest
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.utils import configure_estimator_categorical_features


# ==============================================================================
# Tests: unsupported estimator (no-op)
# ==============================================================================
def test_unsupported_estimator_returns_fit_kwargs_unchanged():
    """
    Test that an unsupported estimator (e.g. LinearRegression) returns
    fit_kwargs unchanged with no side effects.
    """
    estimator = LinearRegression()
    fit_kwargs = {'some_arg': 123}
    result = configure_estimator_categorical_features(
        estimator=estimator,
        categorical_features_names_in_=['cat_col'],
        X_train_features_names_out_=['lag_1', 'cat_col'],
        fit_kwargs=fit_kwargs
    )
    assert result == {'some_arg': 123}


# ==============================================================================
# Tests: no categorical features (reset behavior)
# ==============================================================================
def test_no_categoricals_returns_fit_kwargs_unchanged():
    """
    Test that when categorical_features_names_in_ is None, fit_kwargs is
    returned unchanged.
    """
    fit_kwargs = {'some_arg': 1}
    result = configure_estimator_categorical_features(
        estimator=LGBMRegressor(verbose=-1),
        categorical_features_names_in_=None,
        X_train_features_names_out_=['lag_1', 'lag_2'],
        fit_kwargs=fit_kwargs
    )
    assert result == {'some_arg': 1}


def test_empty_categoricals_returns_fit_kwargs_unchanged():
    """
    Test that when categorical_features_names_in_ is an empty list, fit_kwargs
    is returned unchanged.
    """
    fit_kwargs = {}
    result = configure_estimator_categorical_features(
        estimator=LGBMRegressor(verbose=-1),
        categorical_features_names_in_=[],
        X_train_features_names_out_=['lag_1', 'lag_2'],
        fit_kwargs=fit_kwargs
    )
    assert result == {}


def test_xgboost_reset_feature_types_when_no_categoricals():
    """
    Test that XGBRegressor's feature_types is reset to None when no categorical
    features are present (simulates re-fit without categoricals).
    """
    estimator = XGBRegressor(feature_types=['c', 'q'], enable_categorical=True)
    configure_estimator_categorical_features(
        estimator=estimator,
        categorical_features_names_in_=None,
        X_train_features_names_out_=['cat_col', 'num_col'],
        fit_kwargs={}
    )
    assert estimator.get_params()['feature_types'] is None
    assert estimator.get_params()['enable_categorical'] is False


def test_histgbr_reset_categorical_features_when_no_categoricals():
    """
    Test that HistGradientBoostingRegressor's categorical_features is reset to
    'from_dtype' when no categorical features are present.
    """
    estimator = HistGradientBoostingRegressor(categorical_features=[0])
    configure_estimator_categorical_features(
        estimator=estimator,
        categorical_features_names_in_=None,
        X_train_features_names_out_=['cat_col', 'num_col'],
        fit_kwargs={}
    )
    assert estimator.get_params()['categorical_features'] == 'from_dtype'


# ==============================================================================
# Tests: LightGBM
# ==============================================================================
def test_lightgbm_categorical_feature_added_to_fit_kwargs():
    """
    Test that LGBMRegressor gets categorical_feature as a list of column
    indices in fit_kwargs.
    """
    estimator = LGBMRegressor(verbose=-1)
    features = ['lag_1', 'lag_2', 'cat_a', 'num_1', 'cat_b']
    result = configure_estimator_categorical_features(
        estimator=estimator,
        categorical_features_names_in_=['cat_a', 'cat_b'],
        X_train_features_names_out_=features,
        fit_kwargs={}
    )
    assert result == {'categorical_feature': [2, 4]}


def test_lightgbm_single_categorical_feature():
    """
    Test that LGBMRegressor works with a single categorical feature.
    """
    estimator = LGBMRegressor(verbose=-1)
    features = ['lag_1', 'exog_cat']
    result = configure_estimator_categorical_features(
        estimator=estimator,
        categorical_features_names_in_=['exog_cat'],
        X_train_features_names_out_=features,
        fit_kwargs={}
    )
    assert result == {'categorical_feature': [1]}


def test_lightgbm_warning_when_overriding_categorical_feature_in_fit_kwargs():
    """
    Test that an IgnoredArgumentWarning is raised when the user already
    had categorical_feature in fit_kwargs and it gets overridden.
    """
    estimator = LGBMRegressor(verbose=-1)
    fit_kwargs = {'categorical_feature': [0]}
    features = ['lag_1', 'cat_a']

    warn_msg = re.escape(
        "The `categorical_feature` argument in `fit_kwargs` is being "
        "overridden by the values detected from `categorical_features`. "
        f"Overridden value: {fit_kwargs['categorical_feature']}."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        result = configure_estimator_categorical_features(
            estimator=estimator,
            categorical_features_names_in_=['cat_a'],
            X_train_features_names_out_=features,
            fit_kwargs=fit_kwargs
        )
    assert result == {'categorical_feature': [1]}


def test_lightgbm_no_warning_when_no_prior_categorical_feature():
    """
    Test that no warning is raised for LightGBM when fit_kwargs does not
    contain categorical_feature.
    """
    import warnings as _warnings
    estimator = LGBMRegressor(verbose=-1)
    features = ['lag_1', 'cat_a']
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        result = configure_estimator_categorical_features(
            estimator=estimator,
            categorical_features_names_in_=['cat_a'],
            X_train_features_names_out_=features,
            fit_kwargs={}
        )
    assert result == {'categorical_feature': [1]}


def test_lightgbm_preserves_existing_fit_kwargs():
    """
    Test that existing fit_kwargs entries are preserved when adding
    categorical_feature for LightGBM.
    """
    estimator = LGBMRegressor(verbose=-1)
    features = ['lag_1', 'cat_a']
    result = configure_estimator_categorical_features(
        estimator=estimator,
        categorical_features_names_in_=['cat_a'],
        X_train_features_names_out_=features,
        fit_kwargs={'feature_name': 'auto'}
    )
    assert result == {'feature_name': 'auto', 'categorical_feature': [1]}


# ==============================================================================
# Tests: XGBoost
# ==============================================================================
def test_xgboost_sets_feature_types_and_enable_categorical():
    """
    Test that XGBRegressor gets feature_types and enable_categorical set
    via set_params with the correct values.
    """
    estimator = XGBRegressor()
    features = ['lag_1', 'cat_a', 'num_1']
    result = configure_estimator_categorical_features(
        estimator=estimator,
        categorical_features_names_in_=['cat_a'],
        X_train_features_names_out_=features,
        fit_kwargs={}
    )
    params = estimator.get_params()
    assert params['feature_types'] == ['q', 'c', 'q']
    assert params['enable_categorical'] is True
    assert result == {}


def test_xgboost_multiple_categoricals():
    """
    Test that XGBRegressor handles multiple categorical features correctly.
    """
    estimator = XGBRegressor()
    features = ['cat_a', 'lag_1', 'cat_b', 'lag_2']
    configure_estimator_categorical_features(
        estimator=estimator,
        categorical_features_names_in_=['cat_a', 'cat_b'],
        X_train_features_names_out_=features,
        fit_kwargs={}
    )
    params = estimator.get_params()
    assert params['feature_types'] == ['c', 'q', 'c', 'q']
    assert params['enable_categorical'] is True


def test_xgboost_warning_when_previous_params_differ_from_defaults():
    """
    Test that an IgnoredArgumentWarning is raised for XGBoost when previous
    feature_types was not None or enable_categorical was not True.
    """
    estimator = XGBRegressor(enable_categorical=True, feature_types=['q', 'q'])
    features = ['lag_1', 'cat_a']

    warn_msg = re.escape(
        "The estimator's `feature_types` and `enable_categorical` "
        "parameters have been set to handle categorical features. "
        "Previous values: feature_types=['q', 'q'], "
        "enable_categorical=True."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        configure_estimator_categorical_features(
            estimator=estimator,
            categorical_features_names_in_=['cat_a'],
            X_train_features_names_out_=features,
            fit_kwargs={}
        )


def test_xgboost_no_warning_when_defaults_already_set():
    """
    Test that no warning is raised for XGBoost when feature_types was None
    and enable_categorical was True (defaults match expected).
    """
    import warnings as _warnings
    estimator = XGBRegressor(enable_categorical=True)
    features = ['lag_1', 'cat_a']
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        configure_estimator_categorical_features(
            estimator=estimator,
            categorical_features_names_in_=['cat_a'],
            X_train_features_names_out_=features,
            fit_kwargs={}
        )


def test_xgboost_fit_kwargs_not_modified():
    """
    Test that XGBoost does not add anything to fit_kwargs (configuration
    is done via set_params).
    """
    estimator = XGBRegressor()
    features = ['lag_1', 'cat_a']
    fit_kwargs = {'some_arg': 42}
    result = configure_estimator_categorical_features(
        estimator=estimator,
        categorical_features_names_in_=['cat_a'],
        X_train_features_names_out_=features,
        fit_kwargs=fit_kwargs
    )
    assert result == {'some_arg': 42}


# ==============================================================================
# Tests: HistGradientBoostingRegressor
# ==============================================================================
def test_histgbr_sets_categorical_features():
    """
    Test that HistGradientBoostingRegressor gets categorical_features set
    via set_params with correct indices.
    """
    estimator = HistGradientBoostingRegressor()
    features = ['lag_1', 'cat_a', 'num_1']
    result = configure_estimator_categorical_features(
        estimator=estimator,
        categorical_features_names_in_=['cat_a'],
        X_train_features_names_out_=features,
        fit_kwargs={}
    )
    assert estimator.get_params()['categorical_features'] == [1]
    assert result == {}


def test_histgbr_warning_when_previous_value_not_default():
    """
    Test that an IgnoredArgumentWarning is raised for HistGradientBoosting
    when the previous categorical_features was not None or 'from_dtype'.
    """
    estimator = HistGradientBoostingRegressor(categorical_features=[0, 2])
    features = ['lag_1', 'cat_a']

    warn_msg = re.escape(
        "The estimator's `categorical_features` parameter has been "
        "set to handle categorical features. Previous value: "
        "`categorical_features=[0, 2]`."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        configure_estimator_categorical_features(
            estimator=estimator,
            categorical_features_names_in_=['cat_a'],
            X_train_features_names_out_=features,
            fit_kwargs={}
        )


@pytest.mark.parametrize(
    'default_value',
    [None, 'from_dtype'],
    ids=lambda v: f'default_value: {v}'
)
def test_histgbr_no_warning_when_previous_value_is_default(default_value):
    """
    Test that no warning is raised for HistGradientBoosting when the
    previous categorical_features was None or 'from_dtype'.
    """
    import warnings as _warnings
    estimator = HistGradientBoostingRegressor(categorical_features=default_value)
    features = ['lag_1', 'cat_a']
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        configure_estimator_categorical_features(
            estimator=estimator,
            categorical_features_names_in_=['cat_a'],
            X_train_features_names_out_=features,
            fit_kwargs={}
        )


def test_histgbr_fit_kwargs_not_modified():
    """
    Test that HistGradientBoosting does not add anything to fit_kwargs
    (configuration is done via set_params).
    """
    estimator = HistGradientBoostingRegressor()
    features = ['lag_1', 'cat_a']
    result = configure_estimator_categorical_features(
        estimator=estimator,
        categorical_features_names_in_=['cat_a'],
        X_train_features_names_out_=features,
        fit_kwargs={'some_arg': 1}
    )
    assert result == {'some_arg': 1}


# ==============================================================================
# Tests: CatBoost
# ==============================================================================
def test_catboost_cat_features_added_to_fit_kwargs():
    """
    Test that CatBoostRegressor gets cat_features as a list of column
    indices in fit_kwargs.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    features = ['lag_1', 'cat_a', 'num_1']
    result = configure_estimator_categorical_features(
        estimator=estimator,
        categorical_features_names_in_=['cat_a'],
        X_train_features_names_out_=features,
        fit_kwargs={}
    )
    assert result == {'cat_features': [1]}


def test_catboost_warning_when_overriding_cat_features_in_fit_kwargs():
    """
    Test that an IgnoredArgumentWarning is raised when the user already
    had cat_features in fit_kwargs and it gets overridden.
    """
    estimator = CatBoostRegressor(verbose=0, allow_writing_files=False)
    fit_kwargs = {'cat_features': [0]}
    features = ['lag_1', 'cat_a']

    warn_msg = re.escape(
        "The `cat_features` argument in `fit_kwargs` is being "
        "overridden by the values detected from `categorical_features`. "
        f"Overridden value: {fit_kwargs['cat_features']}."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        result = configure_estimator_categorical_features(
            estimator=estimator,
            categorical_features_names_in_=['cat_a'],
            X_train_features_names_out_=features,
            fit_kwargs=fit_kwargs
        )
    assert result == {'cat_features': [1]}


# ==============================================================================
# Tests: Pipeline (native categorical features are not configured)
# ==============================================================================
@pytest.mark.parametrize(
    'model',
    [
        LGBMRegressor(verbose=-1),
        CatBoostRegressor(verbose=0, allow_writing_files=False),
        XGBRegressor(),
        HistGradientBoostingRegressor(),
    ],
    ids=['LGBMRegressor', 'CatBoostRegressor', 'XGBRegressor',
         'HistGradientBoostingRegressor']
)
def test_pipeline_is_not_configured_when_categorical_features(model):
    """
    Test that, when the estimator is a Pipeline, native categorical features
    are not configured: `fit_kwargs` is returned unchanged (LightGBM and
    CatBoost) and the params of the last step are not set (XGBoost and
    HistGradientBoosting). The previous steps may transform or reorder the
    columns, so the categorical columns are passed with their ordinal encoding.
    """
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    result = configure_estimator_categorical_features(
        estimator=pipe,
        categorical_features_names_in_=['cat_a'],
        X_train_features_names_out_=['lag_1', 'cat_a'],
        fit_kwargs={'some_arg': 123}
    )

    assert result == {'some_arg': 123}
    model_params = pipe[-1].get_params()
    if isinstance(model, XGBRegressor):
        assert model_params['feature_types'] is None
    if isinstance(model, HistGradientBoostingRegressor):
        assert model_params['categorical_features'] == 'from_dtype'


@pytest.mark.parametrize(
    'categorical_features_names_in_',
    [None, [], ['cat_a']],
    ids=lambda names: f'categorical_features_names_in_: {names}'
)
def test_pipeline_keeps_categorical_params_of_last_step(
    categorical_features_names_in_
):
    """
    Test that, when the estimator is a Pipeline, the categorical params the
    user has set on its last step are kept, with and without categorical
    features: they are neither overridden nor reset.
    """
    pipe_xgboost = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(feature_types=['c', 'q'], enable_categorical=True))
    ])
    pipe_histgbr = Pipeline([
        ('scaler', StandardScaler()),
        ('model', HistGradientBoostingRegressor(categorical_features=[0]))
    ])
    for pipe in (pipe_xgboost, pipe_histgbr):
        result = configure_estimator_categorical_features(
            estimator=pipe,
            categorical_features_names_in_=categorical_features_names_in_,
            X_train_features_names_out_=['lag_1', 'cat_a'],
            fit_kwargs={}
        )
        assert result == {}

    assert pipe_xgboost[-1].get_params()['feature_types'] == ['c', 'q']
    assert pipe_xgboost[-1].get_params()['enable_categorical'] is True
    assert pipe_histgbr[-1].get_params()['categorical_features'] == [0]
