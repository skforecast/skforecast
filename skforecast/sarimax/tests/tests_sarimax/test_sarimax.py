# Unit test fit Sarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.sarimax import Sarimax
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

# Fixtures
from .fixtures_Sarimax import y
from .fixtures_Sarimax import y_lw
from .fixtures_Sarimax import exog
from .fixtures_Sarimax import exog_lw
from .fixtures_Sarimax import exog_predict
from .fixtures_Sarimax import exog_lw_predict
from .fixtures_Sarimax import y_datetime
from .fixtures_Sarimax import exog_datetime
from .fixtures_Sarimax import exog_predict_datetime

# Fixtures numpy
y_numpy = y.to_numpy()
exog_numpy = exog.to_numpy()
exog_predict_numpy = exog_predict.to_numpy()

y_lw_numpy = y_lw.to_numpy()
exog_lw_numpy = exog_lw.to_numpy()
exog_lw_predict_numpy = exog_lw_predict.to_numpy()


def test_decorator_check_fitted():
    """
    Test Sarimax decorator check if regressor is fitted.
    """

    sarimax = Sarimax(order=(1, 1, 1))

    err_msg = re.escape(
        ("Sarimax instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using this method.")
    )
    with pytest.raises(NotFittedError, match = err_msg): 
        sarimax.predict(steps=5)
    with pytest.raises(NotFittedError, match = err_msg): 
        sarimax.append(y=y)
    with pytest.raises(NotFittedError, match = err_msg): 
        sarimax.apply(y=y)
    with pytest.raises(NotFittedError, match = err_msg): 
        sarimax.extend(y=y)
    with pytest.raises(NotFittedError, match = err_msg): 
        sarimax.params()
    with pytest.raises(NotFittedError, match = err_msg): 
        sarimax.summary()


def test_Sarimax_sarimax_params_are_stored_during_initialization():
    """
    Test if `sarimax._sarimax_params` are stored correctly during initialization.
    """

    sarimax = Sarimax()
    results = sarimax._sarimax_params

    expected_params = {
        'order': (1, 0, 0),
        'seasonal_order': (0, 0, 0, 0),
        'trend': None,
        'measurement_error': False,
        'time_varying_regression': False,
        'mle_regression': True,
        'simple_differencing': False,
        'enforce_stationarity': True,
        'enforce_invertibility': True,
        'hamilton_representation': False,
        'concentrate_scale': False,
        'trend_offset': 1,
        'use_exact_diffuse': False,
        'dates': None,
        'freq': None,
        'missing': 'none',
        'validate_specification': True,
        'method': 'lbfgs',
        'maxiter': 50,
        'start_params': None,
        'disp': False,
        'sm_init_kwargs': {},
        'sm_fit_kwargs': {},
        'sm_predict_kwargs': {}
    }

    assert isinstance(results, dict)
    assert results == expected_params


def test_Sarimax__repr__(capfd):
    """
    Check information printed by a Sarimax object.
    """

    sarimax = Sarimax(order=(1, 2, 3), seasonal_order=(4, 5, 6, 12))
    expected_out = "Sarimax(1,2,3)(4,5,6)[12]\n"

    print(sarimax)
    out, _ = capfd.readouterr()

    assert out == expected_out


@pytest.mark.parametrize("kwargs, _init_kwargs", 
                         [({'order': (1,1,1), 
                            'sm_init_kwargs': {'order': (2,2,2)}}, 
                           {'order': (1,1,1)}), 
                          ({'order': (2,2,2), 
                            'sm_init_kwargs': {'test': 'fake_param'}}, 
                           {'order': (2,2,2), 
                            'test': 'fake_param'})], 
                         ids = lambda values : f'kwargs, _init_kwargs: {values}')
def test_Sarimax_consolidate_kwargs_init_kwargs(kwargs, _init_kwargs):
    """
    Test if `sarimax._init_kwargs` are correctly consolidate.
    """

    sarimax = Sarimax(**kwargs)
    results = (sarimax._init_kwargs, sarimax.sm_init_kwargs)

    # First expected _init_kwargs, second sm_init_kwargs
    expected = ({
        'order': (1, 0, 0),
        'seasonal_order': (0, 0, 0, 0),
        'trend': None,
        'measurement_error': False,
        'time_varying_regression': False,
        'mle_regression': True,
        'simple_differencing': False,
        'enforce_stationarity': True,
        'enforce_invertibility': True,
        'hamilton_representation': False,
        'concentrate_scale': False,
        'trend_offset': 1,
        'use_exact_diffuse': False,
        'dates': None,
        'freq': None,
        'missing': 'none',
        'validate_specification': True
    },
    kwargs['sm_init_kwargs']
    )
    expected[0].update(_init_kwargs)

    assert all(isinstance(x, dict) for x in results)
    assert results[0] == expected[0]
    assert results[1] == expected[1]


@pytest.mark.parametrize("kwargs, _fit_kwargs", 
                         [({'maxiter': 100, 
                            'sm_fit_kwargs': {'maxiter': 200}}, 
                           {'maxiter': 100}), 
                          ({'maxiter': 200, 
                            'sm_fit_kwargs': {'test': 'fake_param'}}, 
                           {'maxiter': 200, 
                            'test': 'fake_param'})], 
                         ids = lambda values : f'kwargs, _fit_kwargs: {values}')
def test_Sarimax_consolidate_kwargs_fit_kwargs(kwargs, _fit_kwargs):
    """
    Test if `sarimax._fit_kwargs` are correctly consolidate.
    """

    sarimax = Sarimax(**kwargs)
    results = (sarimax._fit_kwargs, sarimax.sm_fit_kwargs)

    # First expected _fit_kwargs, second sm_fit_kwargs
    expected = ({
        'method': 'lbfgs',
        'maxiter': 50,
        'start_params': None,
        'disp': False
    },
    kwargs['sm_fit_kwargs']
    )
    expected[0].update(_fit_kwargs)

    assert all(isinstance(x, dict) for x in results)
    assert results[0] == expected[0]
    assert results[1] == expected[1]


@pytest.mark.parametrize("kwargs, _predict_kwargs", 
                         [({'sm_predict_kwargs': {'test': 'fake_param'}}, 
                           {'test': 'fake_param'})], 
                         ids = lambda values : f'kwargs, _predict_kwargs: {values}')
def test_Sarimax_consolidate_kwargs_predict_kwargs(kwargs, _predict_kwargs):
    """
    Test if `sarimax._predict_kwargs` are correctly consolidate.
    """

    sarimax = Sarimax(**kwargs)
    results = (sarimax._predict_kwargs, sarimax.sm_predict_kwargs)

    # First expected _predict_kwargs, second sm_predict_kwargs
    expected = (
    {},
    kwargs['sm_predict_kwargs']
    )
    expected[0].update(_predict_kwargs)

    assert all(isinstance(x, dict) for x in results)
    assert results[0] == expected[0]
    assert results[1] == expected[1]


def test_Sarimax_create_sarimax():
    """
    Test statsmodels SARIMAX is correctly create with _create_sarimax.
    """

    sarimax = Sarimax(order=(1, 1, 1))
    sarimax._create_sarimax(endog=y)
    sarimax_stats = sarimax.sarimax

    assert isinstance(sarimax_stats, SARIMAX)
    assert sarimax_stats.order == (1, 1, 1)


def test_Sarimax_fit_with_numpy():
    """
    Test Sarimax fit with numpy `y` and `exog`.
    """

    sarimax = Sarimax(order=(1, 1, 1))

    assert sarimax.output_type is None
    assert sarimax.sarimax_res is None
    assert sarimax.fitted == False
    assert sarimax.training_index is None

    sarimax.fit(y=y_numpy, exog=exog_numpy)

    assert sarimax.output_type == 'numpy'
    assert isinstance(sarimax.sarimax_res, SARIMAXResultsWrapper)
    assert sarimax.fitted == True
    assert sarimax.training_index == None


@pytest.mark.parametrize("y, exog", 
                         [(y, exog),
                          (y_datetime, exog_datetime),
                          (y.to_frame(), exog.to_frame()),
                          (y_datetime.to_frame(), exog_datetime.to_frame())], 
                         ids = lambda values : f'y, exog: {type(values)}')
def test_Sarimax_fit_with_pandas(y, exog):
    """
    Test Sarimax fit with pandas `y` and `exog`.
    """

    sarimax = Sarimax(order=(1, 1, 1))

    assert sarimax.output_type is None
    assert sarimax.sarimax_res is None
    assert sarimax.fitted == False
    assert sarimax.training_index is None

    sarimax.fit(y=y, exog=exog)

    assert sarimax.output_type == 'pandas'
    assert isinstance(sarimax.sarimax_res, SARIMAXResultsWrapper)
    assert sarimax.fitted == True
    pd.testing.assert_index_equal(sarimax.training_index, y.index)


def test_Sarimax_predict_with_numpy():
    """
    Test Sarimax predict with numpy `y` and `exog`. It is tested with the 
    skforecast wrapper and the statsmodels SARIMAX.
    """

    # skforecast
    sarimax = Sarimax(order=(1, 1, 1), maxiter=1000, method='cg', disp=False)
    sarimax.fit(y=y_numpy, exog=exog_numpy)
    preds_skforecast = sarimax.predict(steps=10, exog=exog_predict_numpy)

    # statsmodels
    sarimax = SARIMAX(order=(1, 1, 1), endog=y_numpy, exog=exog_numpy)               
    sarimax_res = sarimax.fit(maxiter=1000, method='cg', disp=False)
    preds_statsmodels = sarimax_res.get_forecast(
                            steps = 10, 
                            exog  = exog_predict_numpy
                        ).predicted_mean

    # expected = np.array([0.54251563, 0.57444558, 0.60419802, 0.63585684, 0.67238625,
    #                      0.70992311, 0.76089838, 0.7855481 , 0.73117064, 0.669226  ])

    # np.testing.assert_almost_equal(preds_skforecast, expected)
    # np.testing.assert_almost_equal(preds_statsmodels, expected)
    np.testing.assert_almost_equal(preds_skforecast, preds_statsmodels)


@pytest.mark.parametrize("y, exog, exog_predict", 
                         [(y, exog, exog_predict),
                          (y_datetime, exog_datetime, 
                           exog_predict_datetime),
                          (y.to_frame(), exog.to_frame(), 
                           exog_predict.to_frame()),
                          (y_datetime.to_frame(), exog_datetime.to_frame(), 
                           exog_predict_datetime.to_frame())], 
                         ids = lambda values : f'y, exog, exog_predict: {type(values)}')
def test_Sarimax_predict_with_pandas(y, exog, exog_predict):
    """
    Test Sarimax predict with pandas `y` and `exog`. It is tested with the 
    skforecast wrapper and the statsmodels SARIMAX.
    """

    # skforecast
    sarimax = Sarimax(order=(1, 1, 1), maxiter=1000, method='cg', disp=False)
    sarimax.fit(y=y, exog=exog)
    preds_skforecast = sarimax.predict(steps=10, exog=exog_predict)

    # statsmodels
    sarimax = SARIMAX(order=(1, 1, 1), endog=y, exog=exog)               
    sarimax_res = sarimax.fit(maxiter=1000, method='cg', disp=False)
    preds_statsmodels = sarimax_res.get_forecast(
                            steps = 10, 
                            exog  = exog_predict
                        ).predicted_mean.rename("pred").to_frame()

    # if isinstance(y.index, pd.RangeIndex):
    #     idx = pd.RangeIndex(start=50, stop=60, step=1)
    # else:
    #     idx = pd.date_range(start='2050', periods=10, freq='YE')
    #
    # expected = pd.DataFrame(
    #     data    = np.array([0.54251563, 0.57444558, 0.60419802, 0.63585684, 0.67238625,
    #                         0.70992311, 0.76089838, 0.7855481 , 0.73117064, 0.669226  ]),
    #     index   = idx,
    #     columns = ['pred']
    # )

    # pd.testing.assert_frame_equal(preds_skforecast, expected)
    # pd.testing.assert_frame_equal(preds_statsmodels, expected)
    pd.testing.assert_frame_equal(preds_skforecast, preds_statsmodels)


@pytest.mark.parametrize("y, exog, exog_predict", 
                         [(y_numpy, exog_numpy, exog_predict_numpy),
                          (y_datetime, exog_datetime, exog_predict_datetime),
                          (y_datetime.to_frame(), exog_datetime.to_frame(), 
                           exog_predict_datetime.to_frame())], 
                         ids = lambda values : f'y, exog, exog_predict: {type(values)}')
def test_Sarimax_predict_UserWarning_exog_length_greater_that_steps(y, exog, exog_predict):
    """
    Test Sarimax predict UserWarning is raised with numpy and pandas when length 
    exog is greater than the number of steps.
    """

    # skforecast
    sarimax = Sarimax(order=(1, 1, 1), maxiter=1000, method='cg', disp=False)
    sarimax.fit(y=y, exog=exog)

    steps = 5

    warn_msg = re.escape(
        ("When predicting using exogenous variables, the `exog` parameter "
         "must have the same length as the number of predicted steps. Since "
         "len(exog) > steps, only the first 5 observations are used.")
    )
    with pytest.warns(UserWarning, match = warn_msg):
        sarimax.predict(steps=steps, exog=exog_predict)


def test_Sarimax_predict_interval_with_numpy():
    """
    Test Sarimax predict with confidence intervals using numpy `y` and `exog`.  
    It is tested with the skforecast wrapper and the statsmodels SARIMAX.
    """

    alpha = 0.05

    # skforecast
    sarimax = Sarimax(order=(1, 1, 1), maxiter=1000, method='cg', disp=False)
    sarimax.fit(y=y_numpy, exog=exog_numpy)
    preds_skforecast = sarimax.predict(steps=10, exog=exog_predict_numpy,
                                       return_conf_int=True, alpha=alpha)

    # statsmodels
    sarimax = SARIMAX(order=(1, 1, 1), endog=y_numpy, exog=exog_numpy)               
    sarimax_res = sarimax.fit(maxiter=1000, method='cg', disp=False)
    preds_statsmodels = sarimax_res.get_forecast(
                            steps = 10, 
                            exog  = exog_predict_numpy
                        )
    preds_statsmodels = np.column_stack(
                            [preds_statsmodels.predicted_mean,
                             preds_statsmodels.conf_int(alpha=alpha)]
                        )

    # expected = np.array([[ 0.54251563, -1.19541248,  2.28044374],
    #                      [ 0.57444558, -1.81113178,  2.96002293],
    #                      [ 0.60419802, -2.26035953,  3.46875557],
    #                      [ 0.63585684, -2.62746432,  3.899178  ],
    #                      [ 0.67238625, -2.94147506,  4.28624756],
    #                      [ 0.70992311, -3.22142148,  4.64126771],
    #                      [ 0.76089838, -3.46329933,  4.98509609],
    #                      [ 0.7855481 , -3.7121082 ,  5.28320441],
    #                      [ 0.73117064, -4.02408271,  5.48642398],
    #                      [ 0.669226  , -4.3302976 ,  5.66874961]])

    # np.testing.assert_almost_equal(preds_skforecast, expected)
    # np.testing.assert_almost_equal(preds_statsmodels, expected)
    np.testing.assert_almost_equal(preds_skforecast, preds_statsmodels)


@pytest.mark.parametrize("y, exog, exog_predict", 
                         [(y, exog, exog_predict),
                          (y_datetime, exog_datetime, 
                           exog_predict_datetime),
                          (y.to_frame(), exog.to_frame(), 
                           exog_predict.to_frame()),
                          (y_datetime.to_frame(), exog_datetime.to_frame(), 
                           exog_predict_datetime.to_frame())], 
                         ids = lambda values : f'y, exog, exog_predict: {type(values)}')
def test_Sarimax_predict_interval_with_pandas(y, exog, exog_predict):
    """
    Test Sarimax predict with confidence intervals using pandas `y` and `exog`.  
    It is tested with the skforecast wrapper and the statsmodels SARIMAX.
    """

    alpha = 0.05

    # skforecast
    sarimax = Sarimax(order=(1, 1, 1), maxiter=1000, method='cg', disp=False)
    sarimax.fit(y=y, exog=exog)
    preds_skforecast = sarimax.predict(steps=10, exog=exog_predict,
                                       return_conf_int=True, alpha=alpha)

    # statsmodels
    sarimax = SARIMAX(order=(1, 1, 1), endog=y, exog=exog)               
    sarimax_res = sarimax.fit(maxiter=1000, method='cg', disp=False)
    preds_statsmodels = sarimax_res.get_forecast(
                            steps = 10, 
                            exog  = exog_predict
                        )
    preds_statsmodels = pd.concat((
                            preds_statsmodels.predicted_mean,
                            preds_statsmodels.conf_int(alpha=alpha)),
                            axis = 1
                        )
    preds_statsmodels.columns = ['pred', 'lower_bound', 'upper_bound']

    # if isinstance(y.index, pd.RangeIndex):
    #     idx = pd.RangeIndex(start=50, stop=60, step=1)
    # else:
    #     idx = pd.date_range(start='2050', periods=10, freq='YE')
    
    # expected = pd.DataFrame(
    #     data    = np.array([[ 0.54251563, -1.19541248,  2.28044374],
    #                         [ 0.57444558, -1.81113178,  2.96002293],
    #                         [ 0.60419802, -2.26035953,  3.46875557],
    #                         [ 0.63585684, -2.62746432,  3.899178  ],
    #                         [ 0.67238625, -2.94147506,  4.28624756],
    #                         [ 0.70992311, -3.22142148,  4.64126771],
    #                         [ 0.76089838, -3.46329933,  4.98509609],
    #                         [ 0.7855481 , -3.7121082 ,  5.28320441],
    #                         [ 0.73117064, -4.02408271,  5.48642398],
    #                         [ 0.669226  , -4.3302976 ,  5.66874961]]),
    #     index   = idx,
    #     columns = ['pred', 'lower_bound', 'upper_bound']
    # )

    # pd.testing.assert_frame_equal(preds_skforecast, expected)
    # pd.testing.assert_frame_equal(preds_statsmodels, expected)
    pd.testing.assert_frame_equal(preds_skforecast, preds_statsmodels)


def test_Sarimax_append_refit_False():
    """
    Test Sarimax append with refit `False`. It is tested with the 
    skforecast wrapper and the statsmodels SARIMAX.
    """
    
    refit = False

    # skforecast
    sarimax = Sarimax(order=(1, 1, 1), maxiter=1000, method='cg', disp=False)
    sarimax.fit(y=y_numpy, exog=exog_numpy)
    sarimax.append(
        y     = y_lw_numpy, 
        exog  = exog_lw_numpy,
        refit = refit
    )
    preds_skforecast = sarimax.predict(steps=10, exog=exog_lw_predict_numpy)

    # statsmodels
    sarimax = SARIMAX(order=(1, 1, 1), endog=y_numpy, exog=exog_numpy)               
    sarimax_res = sarimax.fit(maxiter=1000, method='cg', disp=False)
    updated_res = sarimax_res.append(
                      endog = y_lw_numpy, 
                      exog  = exog_lw_numpy,
                      refit = refit
                  )
    preds_statsmodels = updated_res.get_forecast(
                            steps = 10, 
                            exog  = exog_lw_predict_numpy
                        ).predicted_mean

    # expected = np.array([0.89386888, 0.92919515, 0.98327241, 1.02336286, 1.05334974,
    #                      1.08759298, 1.02651694, 0.97164917, 0.91539724, 0.85551674])

    # np.testing.assert_almost_equal(preds_skforecast, expected)
    # np.testing.assert_almost_equal(preds_statsmodels, expected)
    np.testing.assert_almost_equal(preds_skforecast, preds_statsmodels)


def test_Sarimax_append_refit_True():
    """
    Test Sarimax append with refit `True`. It is tested with the 
    skforecast wrapper and the statsmodels SARIMAX.
    """

    refit = True

    # skforecast
    sarimax = Sarimax(order=(1, 0, 1), maxiter=1000, method='cg', disp=False)
    sarimax.fit(y=y_numpy, exog=exog_numpy)
    sarimax.append(
        y     = y_lw_numpy, 
        exog  = exog_lw_numpy,
        refit = refit
    )
    preds_skforecast = sarimax.predict(steps=10, exog=exog_lw_predict_numpy)

    # statsmodels
    sarimax = SARIMAX(order=(1, 0, 1), endog=y_numpy, exog=exog_numpy)               
    sarimax_res = sarimax.fit(maxiter=1000, method='cg', disp=False)
    updated_res = sarimax_res.append(
                      endog = y_lw_numpy, 
                      exog  = exog_lw_numpy,
                      refit = refit,
                      fit_kwargs = {'method': 'cg', 
                                    'maxiter': 1000, 
                                    'start_params': None, 
                                    'disp': True}
                  )
    preds_statsmodels = updated_res.get_forecast(
                            steps = 10, 
                            exog  = exog_lw_predict_numpy
                        ).predicted_mean

    # statsmodels fitting all in the same array (not using append)
    sarimax = SARIMAX(
                  order = (1, 0, 1), 
                  endog = np.append(y_numpy, y_lw_numpy), 
                  exog  = np.append(exog_numpy, exog_lw_numpy)
              )       
    sarimax_res = sarimax.fit(maxiter=1000, method='cg', disp=False)
    preds_statsmodels_2 = updated_res.get_forecast(
                              steps = 10, 
                              exog  = exog_lw_predict_numpy
                          ).predicted_mean

    # expected = np.array([0.73036069, 0.72898841, 0.76386263, 0.78298827, 0.7990083 ,
    #                      0.81640187, 0.78540917, 0.75749361, 0.72887628, 0.69840139])

    # np.testing.assert_almost_equal(preds_skforecast, expected)
    # np.testing.assert_almost_equal(preds_statsmodels, expected)
    # np.testing.assert_almost_equal(preds_statsmodels_2, expected)
    np.testing.assert_almost_equal(preds_skforecast, preds_statsmodels)
    np.testing.assert_almost_equal(preds_skforecast, preds_statsmodels_2)


def test_Sarimax_apply_refit_False():
    """
    Test Sarimax apply with refit `False`. It is tested with the 
    skforecast wrapper and the statsmodels SARIMAX.
    """
    
    refit = False

    # skforecast
    sarimax = Sarimax(order=(1, 1, 1), maxiter=1000, method='cg', disp=False)
    sarimax.fit(y=y_numpy, exog=exog_numpy)
    sarimax.apply(
        y     = y_lw_numpy, 
        exog  = exog_lw_numpy,
        refit = refit
    )
    preds_skforecast = sarimax.predict(steps=10, exog=exog_lw_predict_numpy)

    # statsmodels
    sarimax = SARIMAX(order=(1, 1, 1), endog=y_numpy, exog=exog_numpy)               
    sarimax_res = sarimax.fit(maxiter=1000, method='cg', disp=False)
    updated_res = sarimax_res.apply(
                      endog = y_lw_numpy, 
                      exog  = exog_lw_numpy,
                      refit = refit
                  )
    preds_statsmodels = updated_res.get_forecast(
                            steps = 10, 
                            exog  = exog_lw_predict_numpy
                        ).predicted_mean

    # expected = np.array([0.89386888, 0.92919515, 0.98327241, 1.02336286, 1.05334974,
    #                      1.08759298, 1.02651694, 0.97164917, 0.91539724, 0.85551674])

    # np.testing.assert_almost_equal(preds_skforecast, expected)
    # np.testing.assert_almost_equal(preds_statsmodels, expected)
    np.testing.assert_almost_equal(preds_skforecast, preds_statsmodels)


def test_Sarimax_apply_refit_True():
    """
    Test Sarimax apply with refit `True`. It is tested with the 
    skforecast wrapper and the statsmodels SARIMAX.
    """

    refit = True

    # skforecast
    sarimax = Sarimax(order=(1, 0, 1), maxiter=1000, method='cg', disp=False)
    sarimax.fit(y=y_numpy, exog=exog_numpy)
    sarimax.apply(
        y     = y_lw_numpy, 
        exog  = exog_lw_numpy,
        refit = refit
    )
    preds_skforecast = sarimax.predict(steps=10, exog=exog_lw_predict_numpy)

    # statsmodels
    sarimax = SARIMAX(order=(1, 0, 1), endog=y_numpy, exog=exog_numpy)               
    sarimax_res = sarimax.fit(maxiter=1000, method='cg', disp=False)
    updated_res = sarimax_res.apply(
                      endog = y_lw_numpy, 
                      exog  = exog_lw_numpy,
                      refit = refit,
                      fit_kwargs = {'method': 'cg', 
                                    'maxiter': 1000, 
                                    'start_params': None, 
                                    'disp': True}
                  )
    preds_statsmodels = updated_res.get_forecast(
                            steps = 10, 
                            exog  = exog_lw_predict_numpy
                        ).predicted_mean

    # This is the same as creating a new SARIMAX and fitting with the new data
    sarimax = SARIMAX(
                  order = (1, 0, 1), 
                  endog = y_lw_numpy, 
                  exog  = exog_lw_numpy
              )       
    sarimax_res = sarimax.fit(maxiter=1000, method='cg', disp=False)
    preds_statsmodels_2 = updated_res.get_forecast(
                              steps = 10, 
                              exog  = exog_lw_predict_numpy
                          ).predicted_mean

    # expected = np.array([0.76718596, 0.75795261, 0.79506182, 0.81508184, 0.83164173,
    #                      0.84979957, 0.81751954, 0.78846941, 0.75867965, 0.72695935])

    # np.testing.assert_almost_equal(preds_skforecast, expected)
    # np.testing.assert_almost_equal(preds_statsmodels, expected)
    # np.testing.assert_almost_equal(preds_statsmodels_2, expected)
    np.testing.assert_almost_equal(preds_skforecast, preds_statsmodels)
    np.testing.assert_almost_equal(preds_skforecast, preds_statsmodels_2)


def test_Sarimax_extend():
    """
    Test Sarimax extend. It is tested with the skforecast wrapper and the 
    statsmodels SARIMAX.
    """

    # skforecast
    sarimax = Sarimax(order=(1, 1, 1), maxiter=1000, method='cg', disp=False)
    sarimax.fit(y=y_numpy, exog=exog_numpy)
    sarimax.extend(
        y     = y_lw_numpy, 
        exog  = exog_lw_numpy
    )
    preds_skforecast = sarimax.predict(steps=10, exog=exog_lw_predict_numpy)

    # statsmodels
    sarimax = SARIMAX(order=(1, 1, 1), endog=y_numpy, exog=exog_numpy)
    sarimax_res = sarimax.fit(maxiter=1000, method='cg', disp=False)
    updated_res = sarimax_res.extend(
                      endog = y_lw_numpy, 
                      exog  = exog_lw_numpy
                  )
    preds_statsmodels = updated_res.get_forecast(
                            steps = 10, 
                            exog  = exog_lw_predict_numpy
                        ).predicted_mean

    # expected = np.array([0.89386888, 0.92919515, 0.98327241, 1.02336286, 1.05334974,
    #                      1.08759298, 1.02651694, 0.97164917, 0.91539724, 0.85551674])

    # np.testing.assert_almost_equal(preds_skforecast, expected)
    # np.testing.assert_almost_equal(preds_statsmodels, expected)
    np.testing.assert_almost_equal(preds_skforecast, preds_statsmodels)


def test_Sarimax_set_params():
    """
    Test Sarimax set_params.
    """
    
    sarimax = Sarimax(order=(1, 1, 1))

    sarimax.set_params(
        order         = (2, 2, 2),
        maxiter       = 200,
        sm_fit_kwargs = {'test': 1},
        not_a_param   = 'fake'
    )

    results = sarimax.get_params()

    expected = {
        'order': (2, 2, 2),
        'seasonal_order': (0, 0, 0, 0),
        'trend': None,
        'measurement_error': False,
        'time_varying_regression': False,
        'mle_regression': True,
        'simple_differencing': False,
        'enforce_stationarity': True,
        'enforce_invertibility': True,
        'hamilton_representation': False,
        'concentrate_scale': False,
        'trend_offset': 1,
        'use_exact_diffuse': False,
        'dates': None,
        'freq': None,
        'missing': 'none',
        'validate_specification': True,
        'method': 'lbfgs',
        'maxiter': 200,
        'start_params': None,
        'disp': False,
        'sm_init_kwargs': {},
        'sm_fit_kwargs': {'test': 1},
        'sm_predict_kwargs': {}
    }

    assert results == expected
    assert sarimax.output_type is None
    assert sarimax.sarimax_res is None
    assert sarimax.fitted == False
    assert sarimax.training_index is None


def test_Sarimax_params_numpy():
    """
    Test Sarimax params after fit with numpy `y` and `exog`.
    """

    sarimax_sk = Sarimax(order=(3, 2, 0), disp=0)
    sarimax_sk.fit(y=y_numpy, exog=exog_numpy)

    sarimax_sm = SARIMAX(order=(3, 2, 0), endog=y_numpy, exog=exog_numpy)
    sarimax_res = sarimax_sm.fit(disp=0)

    results_skforecast = sarimax_sk.params()
    results_statsmodels = sarimax_res.params

    # expected = np.array([0.46809671, 0.24916836, 0.45027591, 0.0105501 ])

    # np.testing.assert_almost_equal(results_skforecast, expected)
    # np.testing.assert_almost_equal(results_statsmodels, expected)
    np.testing.assert_almost_equal(results_skforecast, results_statsmodels)


@pytest.mark.parametrize("y, exog", 
                         [(y, exog),
                          (y_datetime, exog_datetime),
                          (y.to_frame(), exog.to_frame()),
                          (y_datetime.to_frame(), exog_datetime.to_frame())], 
                         ids = lambda values : f'y, exog: {type(values)}')
def test_Sarimax_params_pandas(y, exog):
    """
    Test Sarimax params after fit with pandas `y` and `exog`.
    """

    sarimax_sk = Sarimax(order=(3, 2, 0), disp=0)
    sarimax_sk.fit(y=y, exog=exog)

    sarimax_sm = SARIMAX(order=(3, 2, 0), endog=y, exog=exog)
    sarimax_res = sarimax_sm.fit(disp=0)

    results_skforecast = sarimax_sk.params()
    results_statsmodels = sarimax_res.params

    # expected = pd.Series(
    #                data  = np.array([0.46809671, 0.24916836, 0.45027591, 0.0105501 ]),
    #                index = ['exog', 'ar.L1', 'ma.L1', 'sigma2'],
    #            )

    # pd.testing.assert_series_equal(results_skforecast, expected)
    # pd.testing.assert_series_equal(results_statsmodels, expected)
    pd.testing.assert_series_equal(results_skforecast, results_statsmodels)


@pytest.mark.parametrize("y, exog", 
                         [(y_numpy, exog_numpy),
                          (y, exog),
                          (y_datetime, exog_datetime),
                          (y.to_frame(), exog.to_frame()),
                          (y_datetime.to_frame(), exog_datetime.to_frame())], 
                         ids = lambda values : f'y, exog: {type(values)}')
def test_Sarimax_summary(y, exog):
    """
    Test Sarimax params after fit with pandas `y` and `exog`.
    """

    sarimax = Sarimax(order=(1, 0, 1))
    sarimax.fit(y=y, exog=exog)

    # Only 316 characters are checked because the summary includes the date and time of the training
    results = sarimax.summary().as_text()[:316]
    expected = (
        '                               SARIMAX Results                                \n'
        '==============================================================================\n'
        'Dep. Variable:                      y   No. Observations:                   50\n'
        'Model:               SARIMAX(1, 0, 1)   Log Likelihood                  42.591\n'
    )
    
    assert results == expected


def test_Sarimax_get_info_criteria_ValueError_criteria_invalid_value():
    """
    Test Sarimax get_info_criteria ValueError when `criteria` is an invalid value.
    """

    sarimax = Sarimax(order=(1, 0, 1))
    sarimax.fit(y=y, exog=exog)

    criteria = 'not_valid'

    err_msg = re.escape(
        ("Invalid value for `criteria`. Valid options are 'aic', 'bic', "
         "and 'hqic'.")
    )
    with pytest.raises(ValueError, match = err_msg): 
        sarimax.get_info_criteria(criteria=criteria)


def test_Sarimax_get_info_criteria_ValueError_method_invalid_value():
    """
    Test Sarimax get_info_criteria ValueError when `method` is an invalid value.
    """

    sarimax = Sarimax(order=(1, 0, 1))
    sarimax.fit(y=y, exog=exog)

    method = 'not_valid'

    err_msg = re.escape(
        ("Invalid value for `method`. Valid options are 'standard' and "
         "'lutkepohl'.")
    )
    with pytest.raises(ValueError, match = err_msg): 
        sarimax.get_info_criteria(method=method)


@pytest.mark.parametrize("y, exog", 
                         [(y_numpy, exog_numpy),
                          (y, exog),
                          (y_datetime, exog_datetime),
                          (y.to_frame(), exog.to_frame()),
                          (y_datetime.to_frame(), exog_datetime.to_frame())], 
                         ids = lambda values : f'y, exog: {type(values)}')
def test_Sarimax_get_info_criteria(y, exog):
    """
    Test Sarimax get_info_criteria after fit `y` and `exog`.
    """
    
    sarimax_sk = Sarimax(order=(3, 2, 0), disp=0)
    sarimax_sk.fit(y=y, exog=exog)

    sarimax_sm = SARIMAX(order=(3, 2, 0), endog=y, exog=exog)
    sarimax_res = sarimax_sm.fit(disp=0)

    results_skforecast = sarimax_sk.get_info_criteria(criteria='aic', method='standard')
    results_statsmodels = sarimax_res.info_criteria(criteria='aic', method='standard')

    # expected = -42.00040332298444

    # assert results_skforecast == expected
    # assert results_statsmodels == expected
    assert results_skforecast == pytest.approx(results_statsmodels)