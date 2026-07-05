# Unit test MultiEstimatorMixin
# ==============================================================================
import re
import pytest
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from skforecast.base import MultiEstimatorMixin
from skforecast.exceptions import IgnoredArgumentWarning


class CustomFitEstimator(BaseEstimator):
    """
    Minimal estimator whose `fit` method accepts a distinctive keyword argument
    (`custom_arg`) not present in scikit-learn estimators. Used to verify that
    `_check_select_fit_kwargs` selects keyword arguments per estimator according
    to each `fit` signature.
    """

    def fit(self, X, y, custom_arg=None):
        return self


class DummyForecaster(MultiEstimatorMixin):
    """
    Minimal class implementing the MultiEstimatorMixin attribute contract,
    used to test the mixin in isolation. Ids are generated with the real
    `_generate_estimator_ids` method, while `estimators_` are cloned so the
    fitted estimators differ from the original (unfitted) ones.
    `estimator_types` and `estimator_names_` use synthetic values to verify
    index alignment after removal.
    """

    def __init__(self, estimators):
        self.estimators = list(estimators)
        self.estimators_ = [clone(est) for est in self.estimators]
        self.estimator_ids = self._generate_estimator_ids()
        self.estimator_types = [f"type_{i}" for i in range(len(self.estimators))]
        self.estimator_names_ = [f"name_{i}" for i in range(len(self.estimators))]
        self.n_estimators = len(self.estimators)
        self.estimator_params_ = {
            est_id: est.get_params()
            for est_id, est in zip(self.estimator_ids, self.estimators_)
        }


# Test _generate_estimator_ids
# ==============================================================================
def test_generate_estimator_ids_output_when_no_duplicates():
    """
    Test that unique estimators return ids without numeric suffix.
    """
    estimators = [LinearRegression(), Ridge(), RandomForestRegressor()]
    forecaster = DummyForecaster(estimators=estimators)
    results = forecaster._generate_estimator_ids()
    
    expected = [
        'sklearn.LinearRegression',
        'sklearn.Ridge',
        'sklearn.RandomForestRegressor',
    ]

    assert results == expected


def test_generate_estimator_ids_output_when_duplicates():
    """
    Test that duplicate estimators get a numeric suffix appended in order.
    """
    estimators = [
        LinearRegression(),
        LinearRegression(),
        Ridge(),
        LinearRegression(),
    ]
    forecaster = DummyForecaster(estimators=estimators)
    results = forecaster._generate_estimator_ids()

    expected = [
        'sklearn.LinearRegression',
        'sklearn.LinearRegression_2',
        'sklearn.Ridge',
        'sklearn.LinearRegression_3',
    ]

    assert results == expected


@pytest.mark.parametrize(
    'estimators, expected',
    [
        ([LinearRegression()], ['sklearn.LinearRegression']),
        (
            [Ridge(), Ridge()],
            ['sklearn.Ridge', 'sklearn.Ridge_2'],
        ),
    ],
    ids=lambda x: f'estimators: {x}'
)
def test_generate_estimator_ids_output_parametrized(estimators, expected):
    """
    Test ids generated for single and duplicated estimators.
    """
    forecaster = DummyForecaster(estimators=estimators)
    results = forecaster._generate_estimator_ids()

    assert results == expected


# Test get_estimator
# ==============================================================================
def test_get_estimator_raises_KeyError_when_id_not_found():
    """
    Raise KeyError when estimator id is not found.
    """
    forecaster = DummyForecaster(estimators=[LinearRegression()])

    err_msg = re.escape(
        "No estimator with id 'invalid_id'. "
        "Available estimators: ['sklearn.LinearRegression']"
    )
    with pytest.raises(KeyError, match=err_msg):
        forecaster.get_estimator('invalid_id')


def test_get_estimator_returns_correct_estimator():
    """
    Check that get_estimator returns the correct estimator by id.
    """
    estimators = [LinearRegression(), Ridge()]
    forecaster = DummyForecaster(estimators=estimators)

    estimator = forecaster.get_estimator('sklearn.Ridge')

    assert isinstance(estimator, Ridge)
    assert estimator is forecaster.estimators_[1]


def test_get_estimator_returns_correct_estimator_with_suffix():
    """
    Check that get_estimator returns the correct estimator when id has suffix.
    """
    estimators = [LinearRegression(), LinearRegression()]
    forecaster = DummyForecaster(estimators=estimators)

    estimator_1 = forecaster.get_estimator('sklearn.LinearRegression')
    estimator_2 = forecaster.get_estimator('sklearn.LinearRegression_2')

    assert estimator_1 is forecaster.estimators_[0]
    assert estimator_2 is forecaster.estimators_[1]


def test_get_estimator_returns_fitted_estimator():
    """
    Check that the estimator returned by get_estimator is the version stored in
    `estimators_`, not the original one stored in `estimators`.
    """
    forecaster = DummyForecaster(estimators=[LinearRegression()])

    estimator = forecaster.get_estimator(id='sklearn.LinearRegression')

    # estimators_ contains the (fitted) estimators, estimators the originals
    assert estimator is forecaster.estimators_[0]
    assert estimator is not forecaster.estimators[0]


# Test get_estimator_ids
# ==============================================================================
def test_get_estimator_ids_returns_list_of_ids():
    """
    Check that get_estimator_ids returns a list of all estimator ids.
    """
    estimators = [LinearRegression(), Ridge(), RandomForestRegressor()]
    forecaster = DummyForecaster(estimators=estimators)

    ids = forecaster.get_estimator_ids()

    assert ids == [
        'sklearn.LinearRegression',
        'sklearn.Ridge',
        'sklearn.RandomForestRegressor',
    ]
    assert ids is forecaster.estimator_ids


def test_get_estimator_ids_returns_single_id():
    """
    Check that get_estimator_ids returns a list with single id.
    """
    forecaster = DummyForecaster(estimators=[LinearRegression()])

    ids = forecaster.get_estimator_ids()

    assert ids == ['sklearn.LinearRegression']


def test_get_estimator_ids_with_duplicates():
    """
    Check that get_estimator_ids returns unique ids for duplicate estimators.
    """
    estimators = [Ridge(), Ridge(), Ridge()]
    forecaster = DummyForecaster(estimators=estimators)

    ids = forecaster.get_estimator_ids()

    assert ids == ['sklearn.Ridge', 'sklearn.Ridge_2', 'sklearn.Ridge_3']


# Test _check_select_fit_kwargs
# ==============================================================================
def test_check_select_fit_kwargs_TypeError_when_fit_kwargs_not_dict():
    """
    Raise TypeError when `fit_kwargs` is not a dict.
    """
    forecaster = DummyForecaster(estimators=[LinearRegression()])

    err_msg = re.escape(
        "Argument `fit_kwargs` must be a dict. Got <class 'list'>."
    )
    with pytest.raises(TypeError, match=err_msg):
        forecaster._check_select_fit_kwargs(['not_a_dict'])


def test_check_select_fit_kwargs_returns_empty_dict_per_estimator_when_None():
    """
    Check that an empty dict is returned for each estimator when `fit_kwargs`
    is None.
    """
    estimators = [LinearRegression(), Ridge()]
    forecaster = DummyForecaster(estimators=estimators)

    results = forecaster._check_select_fit_kwargs()

    expected = {'sklearn.LinearRegression': {}, 'sklearn.Ridge': {}}

    assert results == expected


def test_check_select_fit_kwargs_drops_sample_weight_with_warning():
    """
    Check that `sample_weight` is removed for every estimator and a warning is
    issued.
    """
    forecaster = DummyForecaster(estimators=[LinearRegression()])

    warn_msg = re.escape(
        "The `sample_weight` argument is ignored. Use `weight_func` to pass "
        "a function that defines the individual weights for each sample "
        "based on its index."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        results = forecaster._check_select_fit_kwargs({'sample_weight': [1, 2]})

    assert results == {'sklearn.LinearRegression': {}}


def test_check_select_fit_kwargs_selects_per_estimator_signature():
    """
    Check that the same `fit_kwargs` is validated against each estimator's `fit`
    signature, keeping the argument only for the estimator that accepts it.
    """
    estimators = [LinearRegression(), CustomFitEstimator()]
    forecaster = DummyForecaster(estimators=estimators)

    warn_msg = re.escape(
        "Argument/s ['custom_arg'] ignored since they are not used by the "
        "estimator's `fit` method."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        results = forecaster._check_select_fit_kwargs({'custom_arg': 1})

    lr_id, custom_id = forecaster.estimator_ids
    expected = {lr_id: {}, custom_id: {'custom_arg': 1}}

    assert results == expected


def test_check_select_fit_kwargs_keeps_kwarg_for_all_accepting_estimators():
    """
    Check that a keyword argument accepted by all estimators is kept for each
    one, including estimators with duplicated ids.
    """
    estimators = [CustomFitEstimator(), CustomFitEstimator()]
    forecaster = DummyForecaster(estimators=estimators)

    results = forecaster._check_select_fit_kwargs({'custom_arg': 1})

    first_id, second_id = forecaster.estimator_ids
    expected = {first_id: {'custom_arg': 1}, second_id: {'custom_arg': 1}}

    assert results == expected


# Test remove_estimators
# ==============================================================================
def test_remove_estimators_raises_KeyError_when_id_not_found():
    """
    Raise KeyError when the estimator id to remove is not found.
    """
    forecaster = DummyForecaster(estimators=[LinearRegression()])

    err_msg = re.escape(
        "No estimator(s) with id '['invalid_id']'. "
        "Available estimators: ['sklearn.LinearRegression']"
    )
    with pytest.raises(KeyError, match=err_msg):
        forecaster.remove_estimators('invalid_id')


def test_remove_estimators_raises_KeyError_when_multiple_ids_not_found():
    """
    Raise KeyError when multiple estimator ids to remove are not found.
    """
    estimators = [LinearRegression(), Ridge()]
    forecaster = DummyForecaster(estimators=estimators)

    err_msg = re.escape(
        "No estimator(s) with id '['invalid_1', 'invalid_2']'. "
        "Available estimators: ['sklearn.LinearRegression', 'sklearn.Ridge']"
    )
    with pytest.raises(KeyError, match=err_msg):
        forecaster.remove_estimators(['invalid_1', 'invalid_2'])


def test_remove_estimators_single_id():
    """
    Check that remove_estimators removes a single estimator by id and keeps
    all contract attributes consistent.
    """
    estimators = [LinearRegression(), Ridge(), RandomForestRegressor()]
    forecaster = DummyForecaster(estimators=estimators)

    forecaster.remove_estimators('sklearn.Ridge')

    assert forecaster.n_estimators == 2
    assert forecaster.estimator_ids == [
        'sklearn.LinearRegression', 'sklearn.RandomForestRegressor'
    ]
    assert len(forecaster.estimators) == 2
    assert len(forecaster.estimators_) == 2
    assert isinstance(forecaster.estimators[0], LinearRegression)
    assert isinstance(forecaster.estimators[1], RandomForestRegressor)
    assert forecaster.estimator_types == ['type_0', 'type_2']
    assert forecaster.estimator_names_ == ['name_0', 'name_2']
    assert list(forecaster.estimator_params_.keys()) == [
        'sklearn.LinearRegression', 'sklearn.RandomForestRegressor'
    ]


def test_remove_estimators_multiple_ids():
    """
    Check that remove_estimators removes multiple estimators by ids.
    """
    estimators = [LinearRegression(), Ridge(), RandomForestRegressor()]
    forecaster = DummyForecaster(estimators=estimators)

    forecaster.remove_estimators(
        ['sklearn.LinearRegression', 'sklearn.RandomForestRegressor']
    )

    assert forecaster.n_estimators == 1
    assert forecaster.estimator_ids == ['sklearn.Ridge']
    assert len(forecaster.estimators) == 1
    assert len(forecaster.estimators_) == 1
    assert isinstance(forecaster.estimators[0], Ridge)
    assert forecaster.estimator_types == ['type_1']
    assert forecaster.estimator_names_ == ['name_1']
    assert list(forecaster.estimator_params_.keys()) == ['sklearn.Ridge']


def test_remove_estimators_with_suffix():
    """
    Check that remove_estimators correctly removes estimator with suffix id.
    """
    estimators = [Ridge(), Ridge(), Ridge()]
    forecaster = DummyForecaster(estimators=estimators)

    forecaster.remove_estimators('sklearn.Ridge_2')

    assert forecaster.n_estimators == 2
    assert forecaster.estimator_ids == ['sklearn.Ridge', 'sklearn.Ridge_3']
    assert len(forecaster.estimators) == 2
    assert len(forecaster.estimators_) == 2
    assert forecaster.estimator_types == ['type_0', 'type_2']
    assert forecaster.estimator_names_ == ['name_0', 'name_2']
    assert list(forecaster.estimator_params_.keys()) == [
        'sklearn.Ridge', 'sklearn.Ridge_3'
    ]


# Test get_estimators_info
# ==============================================================================
def test_get_estimators_info_without_support_attributes():
    """
    Check that get_estimators_info returns id, name, type and params columns
    when the forecaster does not define `estimators_support_exog` or
    `estimators_support_interval`.
    """
    estimators = [LinearRegression(), Ridge()]
    forecaster = DummyForecaster(estimators=estimators)

    results = forecaster.get_estimators_info()

    expected = pd.DataFrame({
        'id': ['sklearn.LinearRegression', 'sklearn.Ridge'],
        'name': ['name_0', 'name_1'],
        'type': ['type_0', 'type_1'],
        'params': [
            str(LinearRegression().get_params()),
            str(Ridge().get_params()),
        ],
    })

    pd.testing.assert_frame_equal(results, expected)


def test_get_estimators_info_with_support_attributes():
    """
    Check that get_estimators_info adds supports_exog and supports_interval
    columns when the forecaster defines the corresponding attributes.
    """
    estimators = [LinearRegression(), Ridge()]
    forecaster = DummyForecaster(estimators=estimators)
    forecaster.estimators_support_exog = ('type_0',)
    forecaster.estimators_support_interval = ('type_1',)

    results = forecaster.get_estimators_info()

    expected = pd.DataFrame({
        'id': ['sklearn.LinearRegression', 'sklearn.Ridge'],
        'name': ['name_0', 'name_1'],
        'type': ['type_0', 'type_1'],
        'supports_exog': [True, False],
        'supports_interval': [False, True],
        'params': [
            str(LinearRegression().get_params()),
            str(Ridge().get_params()),
        ],
    })

    pd.testing.assert_frame_equal(results, expected)


def test_get_estimators_info_after_remove_estimators():
    """
    Check that get_estimators_info stays consistent after removing an estimator.
    """
    estimators = [LinearRegression(), Ridge(), RandomForestRegressor()]
    forecaster = DummyForecaster(estimators=estimators)

    forecaster.remove_estimators('sklearn.Ridge')
    results = forecaster.get_estimators_info()

    assert results['id'].to_list() == [
        'sklearn.LinearRegression', 'sklearn.RandomForestRegressor'
    ]
    assert results['type'].to_list() == ['type_0', 'type_2']
    assert results['params'].to_list() == [
        str(forecaster.estimator_params_['sklearn.LinearRegression']),
        str(forecaster.estimator_params_['sklearn.RandomForestRegressor']),
    ]
