# Unit test select_features
# ==============================================================================
import re
import pytest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures, CalendarFeatures
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.feature_selection import select_features

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
                         ['not_exog_or_autoreg', ['exog', 'invalid']], 
                         ids=lambda so: f'select_only: {so}')
def test_ValueError_select_features_select_only_not_autoreg_exog_calendar_None(select_only):
    """
    Test ValueError is raised in select_features when `select_only` is not 'autoreg',
    'exog', 'calendar' or None.
    """
    forecaster = ForecasterRecursive(
                     estimator = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    err_msg = re.escape(
        "`select_only` must be one or more of the following values: "
        "'autoreg', 'exog', 'calendar', or None."
    )
    with pytest.raises(ValueError, match = err_msg):
        select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = select_only,
        )


@pytest.mark.parametrize("select_only", 
                         [1, False], 
                         ids=lambda so: f'select_only: {so}')
def test_TypeError_select_features_select_only_not_str_list_None(select_only):
    """
    Test TypeError is raised in select_features when `select_only` is not a str,
    a list of str or None.
    """
    forecaster = ForecasterRecursive(
                     estimator = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    err_msg = re.escape(
        "`select_only` must be a str, a list of str, or None."
    )
    with pytest.raises(TypeError, match = err_msg):
        select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = select_only,
        )


def test_ValueError_select_features_when_no_features_to_evaluate():
    """
    Test ValueError is raised in select_features when the group requested in
    `select_only` contains no features (e.g. 'calendar' without calendar features).
    """
    forecaster = ForecasterRecursive(
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
        select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = 'calendar',
        )


@pytest.mark.parametrize("subsample", 
                         [-1, -0.5, 0, 0., 1.1, 2], 
                         ids=lambda ss: f'subsample: {ss}')
def test_ValueError_select_features_subsample_not_greater_0_less_equal_1(subsample):
    """
    Test ValueError is raised in select_features when `subsample` is not in (0, 1].
    """
    forecaster = ForecasterRecursive(
                     estimator = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

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


def test_select_features_when_selector_is_RFE_and_select_only_is_exog_estimator():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog' and estimator is passed to the selector instead
    of forecaster.estimator.
    """
    forecaster = ForecasterRecursive(
                     estimator = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=LinearRegression(), n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'exog',
        verbose     = True,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2']
    assert selected_calendar_features == []


def test_select_features_when_selector_is_RFE_select_only_is_exog_ForecasterRecursive_no_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterRecursive and no window
    features are included.
    """
    forecaster = ForecasterRecursive(
                     estimator = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'exog',
        verbose     = True,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2']
    assert selected_calendar_features == []


def test_select_features_when_selector_is_RFE_select_only_is_exog_ForecasterRecursive_window_features():
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
                     estimator      = LinearRegression(),
                     lags           = 5,
                    window_features = roll_features
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'exog',
        verbose     = True,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == ['roll_mean_3', 'roll_std_5']
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2']
    assert selected_calendar_features == []


def test_select_features_when_selector_is_RFE_select_only_is_autoreg_ForecasterRecursive_no_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'autoreg'. Forecaster is ForecasterRecursive and no window
    features are included.
    """
    forecaster = ForecasterRecursive(
                     estimator = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_lags == [2, 3, 4]
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == []


def test_select_features_when_selector_is_RFE_select_only_is_autoreg_ForecasterRecursive_window_features():
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
                     estimator      = LinearRegression(),
                     lags           = 5,
                     window_features = roll_features
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=4)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_lags == [2, 3, 4]
    assert selected_window_features == ['roll_mean_3']
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == []


def test_select_features_when_selector_is_RFE_select_only_is_None_ForecasterRecursive_no_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is None. Forecaster is ForecasterRecursive and no window
    features are included.
    """
    forecaster = ForecasterRecursive(
                     estimator = LinearRegression(),
                     lags      = 5
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=5)

    warn_msg = re.escape(
        "No autoregressive features have been selected. Since a Forecaster "
        "cannot be created without them, be sure to include at least one "
        "using the `force_inclusion` parameter."
    )
    with pytest.warns(UserWarning, match = warn_msg):
        selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = None,
            verbose     = False,
        )

    assert selected_lags == []
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == []


def test_select_features_when_selector_is_RFE_select_only_is_None_ForecasterRecursive_window_features():
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
                     estimator = LinearRegression(),
                     lags      = 5,
                     window_features = roll_features
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=5)

    warn_msg = re.escape(
        "No autoregressive features have been selected. Since a Forecaster "
        "cannot be created without them, be sure to include at least one "
        "using the `force_inclusion` parameter."
    )
    with pytest.warns(UserWarning, match = warn_msg):
        selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = None,
            verbose     = False,
        )

    assert selected_lags == []
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == []


def test_select_features_when_selector_is_RFE_select_only_autoreg_and_force_inclusion_is_regex():
    """
    Test that select_features returns the expected values when selector is RFE
    select_only is "autoreg" and force_inclusion is regex "^lag_".
    """
    roll_features = RollingFeatures(
                        stats=['mean', 'std'],
                        window_sizes=[3, 5],
                    )
    forecaster = ForecasterRecursive(
                     estimator = LinearRegression(),
                     lags      = 5,
                     window_features = roll_features
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector        = selector,
        forecaster      = forecaster,
        y               = y,
        exog            = exog,
        select_only     = 'autoreg',
        force_inclusion = "^lag_",
        verbose         = False,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == []


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
                        estimator = LinearRegression(),
                        lags      = 5,
                        window_features = roll_features
                    )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector        = selector,
        forecaster      = forecaster,
        y               = y,
        exog            = exog,
        select_only     = None,
        force_inclusion = "^roll_mean",
        verbose         = True,
    )

    assert selected_lags == []
    assert selected_window_features == ['roll_mean_3']
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2']
    assert selected_calendar_features == []


def test_select_features_when_selector_is_RFE_select_force_inclusion_is_list():
    """
    Test that select_features returns the expected values when selector is RFE
    select_only  is None and force_inclusion is list.
    """
    forecaster = ForecasterRecursive(
                     estimator = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector        = selector,
        forecaster      = forecaster,
        y               = y,
        exog            = exog,
        select_only     = None,
        force_inclusion = ['lag_1'],
        verbose         = False,
    )

    assert selected_lags == [1]
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2']
    assert selected_calendar_features == []


def test_select_features_when_selector_is_RFE_select_only_is_exog_ForecasterDirect_no_window_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog'. Forecaster is ForecasterDirect and no window
    features are included.
    """
    forecaster = ForecasterDirect(
                     estimator = LinearRegression(),
                     lags      = 5,
                     steps     = 3
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'exog',
        verbose     = True,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == []
    assert selected_exog == ['exog_1', 'exog_2', 'exog_4']
    assert selected_calendar_features == []


def test_select_features_when_selector_is_RFE_select_only_is_exog_ForecasterDirect_window_features():
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
                     estimator       = LinearRegression(),
                     lags            = 5,
                     window_features = roll_features,
                     steps           = 3
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'exog',
        verbose     = True,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == ['roll_mean_3', 'roll_std_5']
    assert selected_exog == ['exog_1', 'exog_2', 'exog_4']
    assert selected_calendar_features == []


def test_select_features_when_selector_is_RFE_select_only_is_None_ForecasterRecursive_calendar_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is None. Forecaster is ForecasterRecursive with calendar
    features. Selected calendar columns are mapped back to their source features.
    """
    calendar_features = CalendarFeatures(
                            features = ['month', 'week', 'day_of_week', 'hour'],
                            encoding = 'cyclical',
                        )
    forecaster = ForecasterRecursive(
                     estimator         = LinearRegression(),
                     lags              = 5,
                     calendar_features = calendar_features,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=5)

    warn_msg = re.escape(
        "No autoregressive features have been selected. Since a Forecaster "
        "cannot be created without them, be sure to include at least one "
        "using the `force_inclusion` parameter."
    )
    with pytest.warns(UserWarning, match = warn_msg):
        selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = None,
            verbose     = False,
        )

    assert selected_lags == []
    assert selected_window_features == []
    assert selected_exog == ['exog_1', 'exog_2']
    assert selected_calendar_features == ['week', 'hour']


def test_select_features_when_selector_is_RFE_select_only_is_calendar_ForecasterRecursive():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'calendar'. Forecaster is ForecasterRecursive with calendar
    features. Lags and exog are kept unchanged and only calendar features are
    evaluated.
    """
    calendar_features = CalendarFeatures(
                            features = ['month', 'week', 'day_of_week', 'hour'],
                            encoding = 'cyclical',
                        )
    forecaster = ForecasterRecursive(
                     estimator         = LinearRegression(),
                     lags              = 5,
                     calendar_features = calendar_features,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=4)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'calendar',
        verbose     = True,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == ['week', 'day_of_week', 'hour']


def test_select_features_when_selector_is_RFE_select_only_is_list_autoreg_calendar_ForecasterRecursive():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is a list ['autoreg', 'calendar']. Forecaster is
    ForecasterRecursive with calendar features. Exog is kept unchanged because
    it is not in the list of groups to evaluate.
    """
    calendar_features = CalendarFeatures(
                            features = ['month', 'week', 'day_of_week', 'hour'],
                            encoding = 'cyclical',
                        )
    forecaster = ForecasterRecursive(
                     estimator         = LinearRegression(),
                     lags              = 5,
                     calendar_features = calendar_features,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=4)

    warn_msg = re.escape(
        "No autoregressive features have been selected. Since a Forecaster "
        "cannot be created without them, be sure to include at least one "
        "using the `force_inclusion` parameter."
    )
    with pytest.warns(UserWarning, match = warn_msg):
        selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = ['autoreg', 'calendar'],
            verbose     = False,
        )

    assert selected_lags == []
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == ['week', 'day_of_week', 'hour']


def test_select_features_when_selector_is_RFE_select_only_is_calendar_onehot_ForecasterRecursive():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'calendar' with a one-hot encoded calendar feature. The
    selected one-hot columns are mapped back to their single source feature.
    """
    calendar_features = CalendarFeatures(
                            features = ['day_of_week'],
                            encoding = 'onehot',
                        )
    forecaster = ForecasterRecursive(
                     estimator         = LinearRegression(),
                     lags              = 3,
                     calendar_features = calendar_features,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=3)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'calendar',
        verbose     = False,
    )

    assert selected_lags == [1, 2, 3]
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == ['day_of_week']


def test_select_features_when_selector_is_RFE_select_only_is_calendar_and_force_inclusion_is_list():
    """
    Test that select_features returns the expected values when selector is RFE,
    select_only is 'calendar' and force_inclusion is a list of encoded calendar
    columns. The forced columns are mapped back to their source feature.
    """
    calendar_features = CalendarFeatures(
                            features = ['month', 'week', 'day_of_week', 'hour'],
                            encoding = 'cyclical',
                        )
    forecaster = ForecasterRecursive(
                     estimator         = LinearRegression(),
                     lags              = 5,
                     calendar_features = calendar_features,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=2)

    selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
        selector        = selector,
        forecaster      = forecaster,
        y               = y,
        exog            = exog,
        select_only     = 'calendar',
        force_inclusion = ['month_sin', 'month_cos'],
        verbose         = False,
    )

    assert selected_lags == [1, 2, 3, 4, 5]
    assert selected_window_features == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']
    assert selected_calendar_features == ['month', 'week']


def test_select_features_when_selector_is_RFE_select_only_is_None_ForecasterDirect_calendar_features():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is None. Forecaster is ForecasterDirect with calendar
    features. Selected calendar columns are mapped back to their source features.
    """
    calendar_features = CalendarFeatures(
                            features = ['month', 'week', 'day_of_week', 'hour'],
                            encoding = 'cyclical',
                        )
    forecaster = ForecasterDirect(
                     estimator         = LinearRegression(),
                     lags              = 5,
                     steps             = 3,
                     calendar_features = calendar_features,
                 )
    selector = RFE(estimator=forecaster.estimator, n_features_to_select=6)

    warn_msg = re.escape(
        "No autoregressive features have been selected. Since a Forecaster "
        "cannot be created without them, be sure to include at least one "
        "using the `force_inclusion` parameter."
    )
    with pytest.warns(UserWarning, match = warn_msg):
        selected_lags, selected_window_features, selected_exog, selected_calendar_features = select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = None,
            verbose     = False,
        )

    assert selected_lags == []
    assert selected_window_features == []
    assert selected_exog == ['exog_1', 'exog_2', 'exog_4']
    assert selected_calendar_features == ['week', 'hour']
