# Unit test select_features
# ==============================================================================
import re
import pytest
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.feature_selection import select_features
from skforecast.preprocessing import RollingFeatures

# Fixtures
from .fixtures_feature_selection import y_feature_selection as y
from .fixtures_feature_selection import exog_feature_selection as exog


def test_TypeError_select_features_raise_when_forecaster_is_not_supported():
    """
    Test TypeError is raised in select_features when forecaster is not supported.
    """
    forecaster = object()
    selector = RFE(estimator=LinearRegression(), n_features_to_select=3)

    err_msg = re.escape(
        "`forecaster` must be one of the following classes: ['ForecasterRecursive', "
        "'ForecasterDirect']."
    )
    with pytest.raises(TypeError, match = err_msg):
        select_features(
            selector   = selector,
            forecaster = forecaster,
            y          = y,
            exog       = exog,
        )


@pytest.mark.parametrize("select_only", 
                         ['not_exog_or_autoreg', 1, False], 
                         ids=lambda so: f'select_only: {so}')
def test_ValueError_select_features_raise_when_select_only_is_not_autoreg_exog_None(select_only):
    """
    Test ValueError is raised in select_features when `select_only` is not 'autoreg',
    'exog' or None.
    """
    forecaster = ForecasterRecursive(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    err_msg = re.escape(
        "`select_only` must be one of the following values: 'autoreg', 'exog', None."
    )
    with pytest.raises(ValueError, match = err_msg):
        select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = select_only,
        )


@pytest.mark.parametrize("subsample", 
                         [-1, -0.5, 0, 0., 1.1, 2], 
                         ids=lambda ss: f'subsample: {ss}')
def test_ValueError_select_features_raise_when_subsample_is_not_greater_0_less_equal_1(subsample):
    """
    Test ValueError is raised in select_features when `subsample` is not in (0, 1].
    """
    forecaster = ForecasterRecursive(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    err_msg = re.escape(
        "`subsample` must be a number greater than 0 and less than or equal to 1."
    )
    with pytest.raises(ValueError, match = err_msg):
        select_features(
            selector   = selector,
            forecaster = forecaster,
            y          = y,
            exog       = exog,
            subsample  = subsample,
        )


def test_select_features_when_selector_is_RFE_select_only_is_exog_ForecasterAutoreg_no_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterRecursive and no window
    features are included.
    """
    forecaster = ForecasterRecursive(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'exog',
        verbose     = True,
    )

    assert selected_autoreg == [1, 2, 3, 4, 5]
    assert selected_exog == ['exog_1', 'exog_2', 'exog_4']


def test_select_features_when_selector_is_RFE_select_only_is_exog_ForecasterAutoreg_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterRecursive and window
    features are included.
    """
    roll_features = RollingFeatures(
                        stats=['mean', 'std'],
                        window_sizes=[3, 5],
                    )
    forecaster = ForecasterRecursive(
                     regressor      = LinearRegression(),
                     lags           = 5,
                    window_features = roll_features
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'exog',
        verbose     = True,
    )

    assert selected_autoreg == [1, 2, 3, 4, 5, 'roll_mean_3', 'roll_std_5']
    assert selected_exog == ['exog_1', 'exog_2', 'exog_4']



def test_select_features_when_selector_is_RFE_select_only_is_autoreg_ForecasterAutoreg_no_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'autoreg'. Forecaster is ForecasterRecursive and no window
    features are included.
    """
    forecaster = ForecasterRecursive(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_autoreg == [1, 3, 4]
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']


def test_select_features_when_selector_is_RFE_select_only_is_autoreg_ForecasterAutoreg_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'autoreg'. Forecaster is ForecasterRecursive and window
    features are used.
    """
    roll_features = RollingFeatures(
                        stats=['mean', 'std'],
                        window_sizes=[3, 5],
                    )
    forecaster = ForecasterRecursive(
                     regressor      = LinearRegression(),
                     lags           = 5,
                     window_features = roll_features
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=4)

    selected_autoreg, selected_exog = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_autoreg == [1, 3, 4, 'roll_std_5']
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']


def test_select_features_when_selector_is_RFE_select_only_is_None_ForecasterAutoreg_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is None. Forecaster is ForecasterRecursive and window
    features are included.
    """
    roll_features = RollingFeatures(
                        stats=['mean', 'std'],
                        window_sizes=[3, 5],
                    )
    forecaster = ForecasterRecursive(
                     regressor = LinearRegression(),
                     lags      = 5,
                     window_features = roll_features
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=5)

    warn_msg = re.escape(
        ("No autoregressive features have been selected. Since a Forecaster "
         "cannot be created without them, be sure to include at least one "
         "using the `force_inclusion` parameter.")
    )
    with pytest.warns(UserWarning, match = warn_msg):
        selected_autoreg, selected_exog = select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = None,
            verbose     = False,
        )

    assert selected_autoreg == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']



def test_select_features_when_selector_is_RFE_select_only_exog_is_True_and_force_inclusion_is_regex():
    """
    Test that select_features returns the expected values when selector is RFE
    select_only is "exog" and force_inclusion is regex "^exog_3".
    """
    roll_features = RollingFeatures(
                        stats=['mean', 'std'],
                        window_sizes=[3, 5],
                    )
    forecaster = ForecasterRecursive(
                     regressor = LinearRegression(),
                     lags      = 5,
                     window_features = roll_features
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector        = selector,
        forecaster      = forecaster,
        y               = y,
        exog            = exog,
        select_only     = 'exog',
        force_inclusion = "^exog_3",
        verbose         = False,
    )

    assert selected_autoreg == [1, 2, 3, 4, 5, 'roll_mean_3', 'roll_std_5']
    assert selected_exog == ['exog_1', 'exog_2', 'exog_3', 'exog_4']


def test_select_features_when_selector_is_RFE_and_force_inclusion_is_regex():
    """
    Test that select_features returns the expected values when selector is RFE
    select_only is None and force_inclusion is regex "^roll_mean".
    """
    roll_features = RollingFeatures(
                        stats=['mean', 'std'],
                        window_sizes=[3, 5],
                    )
    forecaster = ForecasterRecursive(
                        regressor = LinearRegression(),
                        lags      = 5,
                        window_features = roll_features
                    )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector        = selector,
        forecaster      = forecaster,
        y               = y,
        exog            = exog,
        select_only     = None,
        force_inclusion = "^roll_mean",
        verbose         = True,
    )

    assert selected_autoreg == ['roll_mean_3']
    assert selected_exog == ['exog_1', 'exog_2', 'exog_4']


def test_select_features_when_selector_is_RFE_select_force_inclusion_is_list():
    """
    Test that select_features returns the expected values when selector is RFE
    select_only  is None and force_inclusion is list.
    """
    forecaster = ForecasterRecursive(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector        = selector,
        forecaster      = forecaster,
        y               = y,
        exog            = exog,
        select_only     = None,
        force_inclusion = ['lag_1'],
        verbose         = False,
    )

    assert selected_autoreg == [1]
    assert selected_exog == ['exog_1', 'exog_2', 'exog_4']


def test_select_features_when_selector_is_RFE_select_only_is_exog_ForecasterAutoregDirect_no_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterDirect and no window
    features are included.
    """
    forecaster = ForecasterDirect(
                     regressor = LinearRegression(),
                     lags      = 5,
                     steps     = 3
                 )
    forecaster.window_features = None
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'exog',
        verbose     = True,
    )

    assert selected_autoreg == [1, 2, 3, 4, 5]
    assert selected_exog == ['exog_1', 'exog_2', 'exog_4']


def test_select_features_when_selector_is_RFE_select_only_is_exog_ForecasterAutoregDirect_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterDirect and window
    features are included.
    """
    roll_features = RollingFeatures(
                        stats=['mean', 'std'],
                        window_sizes=[3, 5],
                    )
    forecaster = ForecasterDirect(
                     regressor       = LinearRegression(),
                     lags            = 5,
                     window_features = roll_features,
                     steps           = 3
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'exog',
        verbose     = True,
    )

    assert selected_autoreg == [1, 2, 3, 4, 5, 'roll_mean_3', 'roll_std_5']
    assert selected_exog == ['exog_1', 'exog_2', 'exog_4']
