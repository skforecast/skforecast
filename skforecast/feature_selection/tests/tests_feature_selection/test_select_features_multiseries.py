# Unit test select_features_multiseries
# ==============================================================================
import re
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.feature_selection import select_features_multiseries

# Fixtures
from .fixtures_feature_selection import series
from .fixtures_feature_selection import exog_multiseries as exog


def test_TypeError_select_features_multiseries_raise_when_forecaster_is_not_supported():
    """
    Test TypeError is raised in select_features_multiseries when forecaster is 
    not supported.
    """
    
    err_msg = re.escape(
        "`forecaster` must be one of the following classes: "
        "['ForecasterRecursiveMultiSeries', 'ForecasterDirectMultiVariate']."
    )
    with pytest.raises(TypeError, match = err_msg):
        select_features_multiseries(
            selector   = object(),
            forecaster = object(),
            series     = object(),
            exog       = object(),
        )


@pytest.mark.parametrize("select_only", 
                         ['not_exog_or_autoreg', 1, False], 
                         ids=lambda so: f'select_only: {so}')
def test_ValueError_select_features_multiseries_raise_when_select_only_is_not_autoreg_exog_None(select_only):
    """
    Test ValueError is raised in select_features_multiseries when `select_only` 
    is not 'autoreg', 'exog' or None.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    err_msg = re.escape(
        "`select_only` must be one of the following values: 'autoreg', 'exog', None."
    )
    with pytest.raises(ValueError, match = err_msg):
        select_features_multiseries(
            selector    = selector,
            forecaster  = forecaster,
            series      = object(),
            exog        = object(),
            select_only = select_only,
        )


@pytest.mark.parametrize("subsample", 
                         [-1, -0.5, 0, 0., 1.1, 2], 
                         ids=lambda ss: f'subsample: {ss}')
def test_ValueError_select_features_multiseries_raise_when_subsample_is_not_greater_0_less_equal_1(subsample):
    """
    Test ValueError is raised in select_features_multiseries when `subsample` 
    is not in (0, 1].
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)
    err_msg = re.escape(
        "`subsample` must be a number greater than 0 and less than or equal to 1."
    )
    with pytest.raises(ValueError, match = err_msg):
        select_features_multiseries(
            selector   = selector,
            forecaster = forecaster,
            series     = object(),
            exog       = object(),
            subsample  = subsample,
        )


def test_select_features_multiseries_when_selector_is_RFE_and_select_only_is_exog_regressor():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    and select_only is 'exog' and regressor is passed to the selector instead
    of forecaster.regressor.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = LinearRegression(),
                     lags      = 5,
                     encoding  = 'onehot'
                 )
    selector = RFE(estimator=LinearRegression(), n_features_to_select=2)

    selected_autoreg, selected_exog = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series,
        exog        = exog,
        select_only = 'exog',
        verbose     = False,
    )

    assert selected_autoreg == [1, 2, 3, 4, 5]
    assert selected_exog == ['exog1', 'exog4']


def test_select_features_multiseries_when_selector_is_RFE_and_select_only_is_exog_ForecasterRecursiveMultiSeries_no_window_features():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterRecursiveMultiSeries and 
    no window features are included.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = LinearRegression(),
                     lags      = 5,
                     encoding  = 'ordinal'
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=2)

    selected_autoreg, selected_exog = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series,
        exog        = exog,
        select_only = 'exog',
        verbose     = False,
    )

    assert selected_autoreg == [1, 2, 3, 4, 5]
    assert selected_exog == ['exog1', 'exog4']


def test_select_features_multiseries_when_selector_is_RFE_and_select_only_is_exog_ForecasterRecursiveMultiSeries_window_features():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterRecursiveMultiSeries and 
    window features are included.
    """
    roll_features = RollingFeatures(
                        stats=['mean', 'std'],
                        window_sizes=[3, 5],
                    )
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor       = LinearRegression(),
                     lags            = 5,
                     window_features = roll_features,
                     encoding        = 'ordinal'
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=2)

    selected_autoreg, selected_exog = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series,
        exog        = exog,
        select_only = 'exog',
        verbose     = False,
    )

    assert selected_autoreg == [1, 2, 3, 4, 5, 'roll_mean_3', 'roll_std_5']
    assert selected_exog == ['exog1', 'exog4']


def test_select_features_multiseries_when_selector_is_RFE_and_select_only_is_autoreg_ForecasterRecursiveMultiSeries_no_window_features():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    and select_only is 'autoreg'. Forecaster is ForecasterRecursiveMultiSeries and 
    no window features are included.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = LinearRegression(),
                     lags      = 5,
                     encoding  = 'ordinal'
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=2)

    selected_autoreg, selected_exog = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_autoreg == [4, 5]
    assert selected_exog == ['exog1', 'exog2', 'exog3', 'exog4']


def test_select_features_multiseries_when_selector_is_RFE_and_select_only_is_autoreg_ForecasterRecursiveMultiSeries_window_features():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    and select_only is 'autoreg'. Forecaster is ForecasterRecursiveMultiSeries and 
    window features are included.
    """
    roll_features = RollingFeatures(
                        stats=['mean', 'std'],
                        window_sizes=[3, 5],
                    )
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor       = LinearRegression(),
                     lags            = 5,
                     window_features = roll_features,
                     encoding        = 'ordinal'
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=4)

    selected_autoreg, selected_exog = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_autoreg == [1, 2, 3, 'roll_mean_3']
    assert selected_exog == ['exog1', 'exog2', 'exog3', 'exog4']


def test_select_features_multiseries_when_selector_is_RFE_and_select_only_is_None_no_window_features():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    and select_only is None. Forecaster is ForecasterRecursiveMultiSeries and 
    no window features are included.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     encoding           = 'onehot',
                     transformer_series = StandardScaler()
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    warn_msg = re.escape(
        "No autoregressive features have been selected. Since a Forecaster "
        "cannot be created without them, be sure to include at least one "
        "using the `force_inclusion` parameter."
    )
    with pytest.warns(UserWarning, match = warn_msg):
        selected_autoreg, selected_exog = select_features_multiseries(
            selector    = selector,
            forecaster  = forecaster,
            series      = series,
            exog        = exog,
            select_only = None,
            verbose     = False,
        )

    assert selected_autoreg == []
    assert selected_exog == ['exog1', 'exog3', 'exog4']


def test_select_features_multiseries_when_selector_is_RFE_and_select_only_is_None_window_features():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    and select_only is None. Forecaster is ForecasterRecursiveMultiSeries and 
    window features are included.
    """
    roll_features = RollingFeatures(
                        stats=['mean', 'std'],
                        window_sizes=[3, 5],
                    )
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     window_features    = roll_features,
                     encoding           = 'onehot',
                     transformer_series = StandardScaler()
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series,
        exog        = exog,
        select_only = None,
        verbose     = False,
    )

    assert selected_autoreg == ['roll_std_5']
    assert selected_exog == ['exog1', 'exog4']


def test_select_features_multiseries_when_selector_is_RFE_select_only_exog_is_True_and_force_inclusion_is_regex():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    select_only_exog is True and force_inclusion is regex.
    """
    roll_features = RollingFeatures(
                        stats=['mean', 'std'],
                        window_sizes=[3, 5],
                    )
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor       = LinearRegression(),
                     lags            = 5,
                     window_features = roll_features,
                     encoding        = 'ordinal'
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features_multiseries(
        selector        = selector,
        forecaster      = forecaster,
        series          = series,
        exog            = exog,
        select_only     = 'exog',
        force_inclusion = "^exog_3",
        verbose         = False,
    )

    assert selected_autoreg == [1, 2, 3, 4, 5, 'roll_mean_3', 'roll_std_5']
    assert selected_exog == ['exog1', 'exog3', 'exog4']


def test_select_features_multiseries_when_selector_is_RFE_select_only_exog_is_False_and_force_inclusion_is_list():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    select_only_exog is False and force_inclusion is list.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = LinearRegression(),
                     lags      = 5,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features_multiseries(
        selector        = selector,
        forecaster      = forecaster,
        series          = series,
        exog            = exog,
        select_only     = None,
        force_inclusion = ['lag_1'],
        verbose         = True,
    )

    assert selected_autoreg == [1]
    assert selected_exog == ['exog1', 'exog3', 'exog4']


def test_select_features_when_selector_is_RFE_select_only_is_exog_ForecasterDirectMultiVariate_no_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterDirectMultiVariate and 
    no window features are included.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = LinearRegression(),
                     level     = 'l1',
                     steps     = 3,
                     lags      = 5
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_autoreg == ['l1_lag_1', 'l2_lag_1', 'l2_lag_4']
    assert selected_exog == ['exog1', 'exog2', 'exog3', 'exog4']


def test_select_features_when_selector_is_RFE_select_only_is_exog_ForecasterDirectMultiVariate_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterDirectMultiVariate and 
    window features are included.
    """
    roll_features = RollingFeatures(
                        stats=['mean', 'std'],
                        window_sizes=[3, 5],
                    )
    forecaster = ForecasterDirectMultiVariate(
                     regressor       = LinearRegression(),
                     level           = 'l1',
                     steps           = 3,
                     lags            = 5,
                     window_features = roll_features
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )
    assert selected_autoreg == ['l1_lag_1', 'l1_roll_std_5', 'l2_lag_1']
    assert selected_exog == ['exog1', 'exog2', 'exog3', 'exog4']
