# Unit test select_features_multiseries
# ==============================================================================
import re
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from skforecast.preprocessing import RollingFeatures, CalendarFeatures
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.feature_selection import select_features_multiseries

# Fixtures
from .fixtures_feature_selection import (
    series_wide_range,
    series_dict_range,
    series_wide_datetime,
    series_dict_datetime,
    exog_multiseries as exog,
    exog_feature_selection as exog_datetime,
)


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
                         ['not_exog_or_autoreg', ['exog', 'invalid']], 
                         ids=lambda so: f'select_only: {so}')
def test_ValueError_select_features_multiseries_when_select_only_not_autoreg_exog_calendar_None(select_only):
    """
    Test ValueError is raised in select_features_multiseries when `select_only` 
    is not 'autoreg', 'exog', 'calendar' or None.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    err_msg = re.escape(
        "`select_only` must be one or more of the following values: "
        "'autoreg', 'exog', 'calendar', or None."
    )
    with pytest.raises(ValueError, match = err_msg):
        select_features_multiseries(
            selector    = selector,
            forecaster  = forecaster,
            series      = object(),
            exog        = object(),
            select_only = select_only,
        )


@pytest.mark.parametrize("select_only", 
                         [1, False], 
                         ids=lambda so: f'select_only: {so}')
def test_TypeError_select_features_multiseries_when_select_only_not_str_list_None(select_only):
    """
    Test TypeError is raised in select_features_multiseries when `select_only` 
    is not a str, a list of str or None.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    err_msg = re.escape(
        "`select_only` must be a str, a list of str, or None."
    )
    with pytest.raises(TypeError, match = err_msg):
        select_features_multiseries(
            selector    = selector,
            forecaster  = forecaster,
            series      = object(),
            exog        = object(),
            select_only = select_only,
        )


def test_ValueError_select_features_multiseries_when_no_features_to_evaluate():
    """
    Test ValueError is raised in select_features_multiseries when the group 
    requested in `select_only` contains no features (e.g. 'calendar' without 
    calendar features).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    err_msg = re.escape(
        "No features remain to be evaluated by the selector. The group(s) "
        "requested in `select_only` contain no features. Make sure the "
        "forecaster includes features for the selected group(s)."
    )
    with pytest.raises(ValueError, match = err_msg):
        select_features_multiseries(
            selector    = selector,
            forecaster  = forecaster,
            series      = series_wide_range,
            exog        = exog,
            select_only = 'calendar',
        )


@pytest.mark.parametrize("subsample", 
                         [-1, -0.5, 0, 0., 1.1, 2], 
                         ids=lambda ss: f'subsample: {ss}')
def test_ValueError_select_features_multiseries_when_subsample_not_greater_0_less_equal_1(subsample):
    """
    Test ValueError is raised in select_features_multiseries when `subsample` 
    is not in (0, 1].
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)
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


def test_select_features_multiseries_when_selector_is_RFE_and_select_only_is_exog_estimator():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    and select_only is 'exog' and estimator is passed to the selector instead
    of forecaster.estimator.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = LinearRegression(),
                     lags      = 5,
                     encoding  = 'onehot'
                 )
    selector = RFE(estimator=LinearRegression(), n_features_to_select=2)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_dict_range,
        exog        = exog,
        select_only = 'exog',
        verbose     = False,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == []
    assert selected_exog == ['exog1', 'exog4']
    assert selected_calendar_features == []


def test_select_features_multiseries_when_selector_is_RFE_and_select_only_is_exog_ForecasterRecursiveMultiSeries_no_window_features():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterRecursiveMultiSeries and 
    no window features are included.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = LinearRegression(),
                     lags      = 5,
                     encoding  = 'ordinal'
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=2)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_dict_range,
        exog        = exog,
        select_only = 'exog',
        verbose     = False,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == []
    assert selected_exog == ['exog1', 'exog4']
    assert selected_calendar_features == []


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
                     estimator       = LinearRegression(),
                     lags            = 5,
                     window_features = roll_features,
                     encoding        = 'ordinal'
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=2)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_dict_range,
        exog        = exog,
        select_only = 'exog',
        verbose     = False,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == ['roll_mean_3', 'roll_std_5']
    assert selected_exog == ['exog1', 'exog4']
    assert selected_calendar_features == []


def test_select_features_multiseries_when_selector_is_RFE_and_select_only_is_autoreg_ForecasterRecursiveMultiSeries_no_window_features():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    and select_only is 'autoreg'. Forecaster is ForecasterRecursiveMultiSeries and 
    no window features are included.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = LinearRegression(),
                     lags      = 5,
                     encoding  = 'ordinal'
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=2)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_dict_range,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_lags == [4, 5]
    assert selected_window_features == []
    assert selected_exog == ['exog1', 'exog2', 'exog3', 'exog4']
    assert selected_calendar_features == []


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
                     estimator       = Ridge(alpha=0.1, random_state=123),
                     lags            = 5,
                     window_features = roll_features,
                     encoding        = 'ordinal'
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=4)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_dict_range,
        exog        = exog,
        select_only = 'autoreg',
        subsample   = 0.9,
        verbose     = False,
    )

    assert selected_lags == [3, 4, 5]
    assert selected_window_features == ['roll_std_5']
    assert selected_exog == ['exog1', 'exog2', 'exog3', 'exog4']
    assert selected_calendar_features == []


def test_select_features_multiseries_when_selector_is_RFE_and_select_only_is_None_no_window_features():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    and select_only is None. Forecaster is ForecasterRecursiveMultiSeries and 
    no window features are included.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     encoding           = 'onehot',
                     transformer_series = StandardScaler()
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=2)

    warn_msg = re.escape(
        "No autoregressive features have been selected. Since a Forecaster "
        "cannot be created without them, be sure to include at least one "
        "using the `force_inclusion` parameter."
    )
    with pytest.warns(UserWarning, match = warn_msg):
        selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
            selector    = selector,
            forecaster  = forecaster,
            series      = series_dict_range,
            exog        = exog,
            select_only = None,
            verbose     = False,
        )

    assert selected_lags == []
    assert selected_window_features == []
    assert selected_exog == ['exog1', 'exog4']
    assert selected_calendar_features == []


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
                     estimator          = LinearRegression(),
                     lags               = 5,
                     window_features    = roll_features,
                     encoding           = 'onehot',
                     transformer_series = StandardScaler()
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_dict_range,
        exog        = exog,
        select_only = None,
        verbose     = False,
    )

    assert selected_lags == []
    assert selected_window_features == ['roll_std_5']
    assert selected_exog == ['exog1', 'exog4']
    assert selected_calendar_features == []


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
                     estimator       = LinearRegression(),
                     lags            = 5,
                     window_features = roll_features,
                     encoding        = 'ordinal'
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector        = selector,
        forecaster      = forecaster,
        series          = series_dict_range,
        exog            = exog,
        select_only     = 'exog',
        force_inclusion = "^exog_3",
        verbose         = False,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == ['roll_mean_3', 'roll_std_5']
    assert selected_exog == ['exog1', 'exog3', 'exog4']
    assert selected_calendar_features == []


def test_select_features_multiseries_when_selector_is_RFE_select_only_exog_is_False_and_force_inclusion_is_list():
    """
    Test that select_features_multiseries returns the expected values when selector is RFE
    select_only_exog is False and force_inclusion is list.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = LinearRegression(),
                     lags      = 5,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector        = selector,
        forecaster      = forecaster,
        series          = series_dict_range,
        exog            = exog,
        select_only     = None,
        force_inclusion = ['lag_1'],
        verbose         = True,
    )

    assert selected_lags == [1, 4]
    assert selected_window_features == []
    assert selected_exog == ['exog1', 'exog4']
    assert selected_calendar_features == []


@pytest.mark.parametrize("lags", 
                         [{'l1': None, 'l2': 5}, {'l1': [], 'l2': 5}],
                         ids = lambda lags: f'lags: {lags}')
def test_select_features_when_RFE_select_only_exog_ForecasterDirectMultiVariate_lags_dict(lags):
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterDirectMultiVariate 
    and lags is a dictionary.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator = LinearRegression(),
                     level     = 'l1',
                     steps     = 3,
                     lags      = lags
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_wide_range,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_lags == {'l1': [], 'l2': [2, 3, 4]}
    assert selected_window_features == []
    assert selected_exog == ['exog1', 'exog2', 'exog3', 'exog4']
    assert selected_calendar_features == []


def test_select_features_when_selector_is_RFE_select_only_is_exog_ForecasterDirectMultiVariate_no_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterDirectMultiVariate and 
    no window features are included.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator = LinearRegression(),
                     level     = 'l1',
                     steps     = 3,
                     lags      = 5
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_wide_range,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_lags == {'l1': [3, 5], 'l2': [3]}
    assert selected_window_features == []
    assert selected_exog == ['exog1', 'exog2', 'exog3', 'exog4']
    assert selected_calendar_features == []


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
                     estimator       = LinearRegression(),
                     level           = 'l1',
                     steps           = 3,
                     lags            = 5,
                     window_features = roll_features
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_wide_range,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_lags == {'l1': [3, 5], 'l2': [3]}
    assert selected_window_features == []
    assert selected_exog == ['exog1', 'exog2', 'exog3', 'exog4']
    assert selected_calendar_features == []


def test_select_features_multiseries_when_selector_is_RFE_select_only_is_calendar_ForecasterRecursiveMultiSeries():
    """
    Test that select_features_multiseries returns the expected values when selector
    is RFE and select_only is 'calendar'. Forecaster is ForecasterRecursiveMultiSeries
    with calendar features. Lags and exog are kept unchanged and only calendar
    features are evaluated.
    """
    calendar_features = CalendarFeatures(
                            features = ['month', 'week', 'day_of_week', 'hour'],
                            encoding = 'cyclical',
                        )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     encoding           = 'ordinal',
                     calendar_features  = calendar_features,
                     transformer_series = StandardScaler()
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=4)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_dict_datetime,
        exog        = exog_datetime,
        select_only = 'calendar',
        verbose     = False,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == ['week', 'day_of_week', 'hour']


def test_select_features_multiseries_when_selector_is_RFE_select_only_is_None_ForecasterRecursiveMultiSeries_calendar_features():
    """
    Test that select_features_multiseries returns the expected values when selector
    is RFE and select_only is None. Forecaster is ForecasterRecursiveMultiSeries
    with calendar features and onehot encoding. Selected calendar columns are
    mapped back to their source features.
    """
    calendar_features = CalendarFeatures(
                            features = ['month', 'week', 'day_of_week', 'hour'],
                            encoding = 'cyclical',
                        )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     encoding           = 'onehot',
                     calendar_features  = calendar_features,
                     transformer_series = StandardScaler()
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=6)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_dict_datetime,
        exog        = exog_datetime,
        select_only = None,
        verbose     = False,
    )

    assert selected_lags == [1]
    assert selected_window_features == []
    assert selected_exog == ['exog_1', 'exog_2']
    assert selected_calendar_features == ['week', 'hour']


def test_select_features_multiseries_when_selector_is_RFE_select_only_is_list_autoreg_calendar_ForecasterRecursiveMultiSeries():
    """
    Test that select_features_multiseries returns the expected values when selector
    is RFE and select_only is a list ['autoreg', 'calendar']. Forecaster is
    ForecasterRecursiveMultiSeries with calendar features. Exog is kept unchanged
    because it is not in the list of groups to evaluate.
    """
    calendar_features = CalendarFeatures(
                            features = ['month', 'week', 'day_of_week', 'hour'],
                            encoding = 'cyclical',
                        )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     encoding           = 'ordinal',
                     calendar_features  = calendar_features,
                     transformer_series = StandardScaler()
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=4)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_dict_datetime,
        exog        = exog_datetime,
        select_only = ['autoreg', 'calendar'],
        verbose     = False,
    )

    assert selected_lags == [1]
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == ['week', 'hour']


def test_select_features_multiseries_when_selector_is_RFE_select_only_is_calendar_and_force_inclusion_is_list():
    """
    Test that select_features_multiseries returns the expected values when selector
    is RFE, select_only is 'calendar' and force_inclusion is a list of encoded
    calendar columns. The forced columns are mapped back to their source feature.
    """
    calendar_features = CalendarFeatures(
                            features = ['month', 'week', 'day_of_week', 'hour'],
                            encoding = 'cyclical',
                        )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     encoding           = 'ordinal',
                     calendar_features  = calendar_features,
                     transformer_series = StandardScaler()
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=2)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector        = selector,
        forecaster      = forecaster,
        series          = series_dict_datetime,
        exog            = exog_datetime,
        select_only     = 'calendar',
        force_inclusion = ['month_sin', 'month_cos'],
        verbose         = False,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == ['month', 'week']


def test_select_features_multiseries_when_selector_is_RFE_select_only_is_None_ForecasterDirectMultiVariate_calendar_features():
    """
    Test that select_features_multiseries returns the expected values when selector
    is RFE and select_only is None. Forecaster is ForecasterDirectMultiVariate
    with calendar features. Selected calendar columns are mapped back to their
    source features and lags are returned as a dict per series.
    """
    calendar_features = CalendarFeatures(
                            features = ['month', 'week', 'day_of_week', 'hour'],
                            encoding = 'cyclical',
                        )
    forecaster = ForecasterDirectMultiVariate(
                     estimator         = LinearRegression(),
                     level             = 'l1',
                     steps             = 3,
                     lags              = 5,
                     calendar_features = calendar_features,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=8)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_wide_datetime,
        exog        = exog_datetime,
        select_only = None,
        verbose     = False,
    )

    assert selected_lags == {'l1': [], 'l2': [4]}
    assert selected_window_features == []
    assert selected_exog == ['exog_1', 'exog_2', 'exog_4']
    assert selected_calendar_features == ['week', 'day_of_week', 'hour']


def test_select_features_multiseries_when_selector_is_RFE_select_only_is_calendar_ForecasterDirectMultiVariate():
    """
    Test that select_features_multiseries returns the expected values when selector
    is RFE and select_only is 'calendar'. Forecaster is ForecasterDirectMultiVariate
    with calendar features. Lags and exog are kept unchanged and only calendar
    features are evaluated.
    """
    calendar_features = CalendarFeatures(
                            features = ['month', 'week', 'day_of_week', 'hour'],
                            encoding = 'cyclical',
                        )
    forecaster = ForecasterDirectMultiVariate(
                     estimator         = LinearRegression(),
                     level             = 'l1',
                     steps             = 3,
                     lags              = 5,
                     calendar_features = calendar_features,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=4)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features_multiseries(
        selector    = selector,
        forecaster  = forecaster,
        series      = series_wide_datetime,
        exog        = exog_datetime,
        select_only = 'calendar',
        verbose     = False,
    )

    assert selected_lags == {'l1': [1, 2, 3, 4, 5], 'l2': [1, 2, 3, 4, 5]}
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == ['week', 'day_of_week', 'hour']
