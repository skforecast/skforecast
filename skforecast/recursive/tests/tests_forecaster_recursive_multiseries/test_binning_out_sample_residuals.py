# Unit test _binning_out_sample_residuals ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from skforecast.exceptions import ResidualsUsageWarning
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                       'l2': pd.Series(np.arange(10))})


@pytest.mark.parametrize("encoding", 
                         ['ordinal', 'onehot', 'ordinal_category'], 
                         ids=lambda encoding: f'encoding: {encoding}')
def test_binning_out_sample_residuals_when_residuals_length_is_less_than_10000_and_no_append(encoding):
    """
    Test residuals stored when new residuals length is less than 10000 and append
    is False.
    """
    rng = np.random.default_rng(12345)
    series = pd.DataFrame({"l1": rng.normal(10, 3, 20), "l2": rng.normal(10, 3, 20)})
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, binner_kwargs={"n_bins": 3}
    )
    forecaster.fit(series=series)
    y_true = {"l1": rng.normal(10, 3, 20), "l2": rng.normal(10, 3, 20)}
    y_pred = {"l1": rng.normal(10, 3, 20), "l2": rng.normal(10, 3, 20)}

    forecaster.out_sample_residuals_ = {}
    forecaster.out_sample_residuals_by_bin_ = {}
    results = forecaster._binning_out_sample_residuals(
        level="l1", y_true=y_true["l1"], y_pred=y_pred["l1"]
    )

    expected_out_sample = np.array(
        [1.60962553,  1.71263736,  9.19708889, -2.73386424,  1.51464604,
        -3.32699998,  3.5987987 ,  0.13608979, -2.06018676, -2.72988393,
        1.63221968, -0.57130461, -1.08106429, -1.45886519,  0.91789988,
        -2.19933831,  1.97853178, -2.56217282,  3.69683321, -4.86497532]
    )
    expected_out_sample_by_bin = {
        0: np.array([
            1.60962553, 1.71263736, 9.19708889, 1.51464604, 3.5987987,
            1.63221968, -2.19933831, 1.97853178, -2.56217282, 3.69683321,
        ]),
        1: np.array([
            -3.32699998,
        ]),
        2: np.array([
            -2.73386424, 0.13608979, -2.06018676, -2.72988393, -0.57130461,
            -1.08106429, -1.45886519, 0.91789988, -4.86497532,
        ])
    }
    
    np.testing.assert_array_almost_equal(results[0], expected_out_sample)
    assert results[1].keys() == expected_out_sample_by_bin.keys()
    assert results[1].keys() == expected_out_sample_by_bin.keys()
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(
            results[1][k], expected_out_sample_by_bin[k]
        )


def test_binning_out_sample_residuals_when_residuals_length_is_less_than_10000_and_no_append_encoding_None():
    """
    Test residuals stored when new residuals length is less than 10000 and append
    is False and encoding is None.
    """
    rng = np.random.default_rng(12345)
    series = pd.DataFrame({"l1": rng.normal(10, 3, 20), "l2": rng.normal(10, 3, 20)})
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=None, binner_kwargs={"n_bins": 3}
    )
    forecaster.fit(series=series)
    y_true = {"_unknown_level": pd.Series(rng.normal(10, 3, 20))}
    y_pred = {"_unknown_level": pd.Series(rng.normal(10, 3, 20))}

    forecaster.out_sample_residuals_ = {}
    forecaster.out_sample_residuals_by_bin_ = {}
    results = forecaster._binning_out_sample_residuals(
        level="_unknown_level", y_true=y_true["_unknown_level"], y_pred=y_pred["_unknown_level"]
    )

    expected_out_sample = np.array(
        [ 1.15252117, -1.81458426,  5.05451064, -0.57203687,  4.72280084,
         -0.75286693, -1.65697481,  7.50482204, -1.7679251 ,  4.37782366,
          4.02311098,  4.1187269 ,  1.56932008,  0.0070832 ,  5.05247358,
         -1.84063415, -3.96576256, -3.85592894,  3.24784139, -4.9446084]
    )
    expected_out_sample_by_bin = {
        0: np.array([
            5.05451064,  4.72280084, -0.75286693,  7.50482204,  4.37782366,
            4.02311098,  4.1187269 ,  5.05247358, -1.84063415, -3.85592894,
            3.24784139
        ]),
        1: np.array([
            1.15252117, -0.57203687,  1.56932008
        ]),
        2: np.array([
            -1.81458426, -1.65697481, -1.7679251 ,  0.0070832 , -3.96576256,
            -4.9446084
        ])
    }
    
    np.testing.assert_array_almost_equal(results[0], expected_out_sample)
    assert results[1].keys() == expected_out_sample_by_bin.keys()
    assert results[1].keys() == expected_out_sample_by_bin.keys()
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(
            results[1][k], expected_out_sample_by_bin[k]
        )


@pytest.mark.parametrize("encoding", ['ordinal', 'onehot', 'ordinal_category'], 
                         ids=lambda encoding: f'encoding: {encoding}')
def test_binning_out_sample_residuals_when_residuals_length_is_less_than_10000_and_append(encoding):
    """
    Test residuals stored when new residuals length is less than 10_000 and append is True.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, binner_kwargs={'n_bins': 3}
    )
    forecaster.fit(series=series)
    y_true = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([2, 3, 4, 5, 6])}
    y_pred = {'l1': np.array([0, 1, 2, 3, 4]), 'l2': np.array([0, 1, 2, 3, 4])}
    
    forecaster.out_sample_residuals_ = {}
    forecaster.out_sample_residuals_by_bin_ = {}

    warn_msg = re.escape(
        f"The following bins of level 'l1' have no out of sample residuals: [1, 2]. "
        f"No predicted values fall in the interval {[forecaster.binner_intervals_['l1'][bin] for bin in [1, 2]]}. "
        f"Empty bins will be filled with a random sample of residuals."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        residuals = forecaster._binning_out_sample_residuals(
            level="l1", y_true=y_true["l1"], y_pred=y_pred["l1"]
        )
    forecaster.out_sample_residuals_ = {"l1": residuals[0]}
    forecaster.out_sample_residuals_by_bin_ = {"l1": residuals[1]}

    y_true = {'l1': np.array([2, 3, 4, 5, 6]), 'l2': np.array([2, 3, 4, 5, 6])}
    y_pred = {'l1': np.array([0, 1, 2, 3, 4]), 'l2': np.array([0, 1, 2, 3, 4])}
    results = forecaster._binning_out_sample_residuals(
        level="l1", y_true=y_true["l1"], y_pred=y_pred["l1"], append=True
    )

    expected_out_sample = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    expected_out_sample_by_bin = {
        0: np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]), 
        1: np.array([1, 1, 1, 1, 1]),
        2: np.array([1, 1, 1, 1, 1])
    }

    np.testing.assert_array_almost_equal(results[0], expected_out_sample)
    assert results[1].keys() == expected_out_sample_by_bin.keys()
    assert results[1].keys() == expected_out_sample_by_bin.keys()
    for k in results[1].keys():
        np.testing.assert_array_almost_equal(
            results[1][k], expected_out_sample_by_bin[k]
        )


@pytest.mark.parametrize("encoding", 
                         ['ordinal', 'onehot', 'ordinal_category'], 
                         ids=lambda encoding: f'encoding: {encoding}')
def test_binning_out_sample_residuals_when_residuals_length_is_greater_than_10000(encoding):
    """
    Test len residuals stored when its length is greater than 10000.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, binner_kwargs={'n_bins': 2}
    )
    forecaster.fit(series=series)
    y_true = {'l1': np.ones(20_000)}
    y_pred = {'l1': np.concatenate([np.ones(10_000) + 1, np.ones(10_000) + 2])}
    
    forecaster.out_sample_residuals_ = {}
    forecaster.out_sample_residuals_by_bin_ = {}
    results = forecaster._binning_out_sample_residuals(
        level="l1", y_true=y_true["l1"], y_pred=y_pred["l1"]
    )

    assert len(results[0]) == 10_000
    for v in results[1].values():
        assert len(v) == 5_000


def test_binning_out_sample_residuals_when_residuals_length_is_greater_than_10000_encoding_None():
    """
    Test len residuals stored when its length is greater than 10_000
    and encoding is None.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=None,  binner_kwargs={'n_bins': 2}
    )
    forecaster.fit(series=series)
    y_true = {'_unknown_level': np.ones(20_000)}
    y_pred = {'_unknown_level': np.concatenate([np.ones(10_000) + 1, np.ones(10_000) + 2])}
    
    forecaster.out_sample_residuals_ = {}
    forecaster.out_sample_residuals_by_bin_ = {}
    results = forecaster._binning_out_sample_residuals(
        level="_unknown_level", y_true=y_true["_unknown_level"], y_pred=y_pred["_unknown_level"]
    )

    assert len(results[0]) == 10_000
    for v in results[1].values():
        assert len(v) == 5_000


def test_binning_out_sample_residuals_when_residuals_length_is_greater_than_10000_and_append():
    """
    Test residuals stored when new residuals length is greater than 10000 and
    append is True.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding='ordinal',  binner_kwargs={'n_bins': 2}
    )
    forecaster.fit(series=series)
    y_true = {'l1': np.ones(5_000)}
    y_pred = {'l1': np.concatenate([np.ones(2_500) + 1, np.ones(2_500) + 2])}

    forecaster.out_sample_residuals_ = {}
    forecaster.out_sample_residuals_by_bin_ = {}
    results = forecaster._binning_out_sample_residuals(
        level="l1", y_true=y_true["l1"], y_pred=y_pred["l1"]
    )
    
    y_true = {'l1': np.ones(20_000)}
    y_pred = {'l1': np.concatenate([np.ones(10_000) + 1, np.ones(10_000) + 2])}
    results = forecaster._binning_out_sample_residuals(
        level="l1", y_true=y_true["l1"], y_pred=y_pred["l1"], append=True
    )

    assert len(results[0]) == 10_000
    for v in results[1].values():
        assert len(v) == 5_000


@pytest.mark.parametrize("differentiation", 
                         [1, {'l1': 1, 'l2': 1, '_unknown_level': 1}], 
                         ids=lambda diff: f'differentiation: {diff}')
def test_forecaster_binning_out_sample_residuals_when_transformer_series_and_differentiation(differentiation):
    """
    Test _binning_out_sample_residuals when forecaster has transformer_series and differentiation.
    Stored should equivalent to residuals calculated manually if transformer_series and
    differentiation are applied to `y_true` and `y_pred` before calculating residuals.
    """
    series_train = {
        'l1': pd.Series(
            np.array([-1.42382504,  1.26372846, -0.87066174, -0.25917323, -0.07534331,
                      -0.74088465, -1.3677927 ,  0.6488928 ,  0.36105811, -1.95286306,
                       2.34740965,  0.96849691, -0.75938718,  0.90219827, -0.46695317,
                      -0.06068952,  0.78884434, -1.25666813,  0.57585751,  1.39897899]),
            index = pd.date_range(start='1-1-2018', periods=20, freq='D')
        ),
        'l2': pd.Series(
            np.array([1.32229806, -0.29969852,  0.90291934, -1.62158273, -0.15818926,
                      0.44948393, -1.34360107, -0.08168759,  1.72473993,  2.61815943,
                      0.77736134,  0.8286332 , -0.95898831, -1.20938829, -1.41229201,
                      0.54154683,  0.7519394 , -0.65876032, -1.22867499,  0.25755777]),
            index = pd.date_range(start='1-1-2018', periods=20, freq='D')
        )
    }
    y_true  = {
        'l1': np.array([ 0.31290292, -0.13081169,  1.26998312, -0.09296246, -0.06615089]),
        'l2': np.array([-1.10821447,  0.13595685,  1.34707776,  0.06114402,  0.0709146 ])
    }
    y_pred = {
        'l1': np.array([0.43365454, 0.27748366, 0.53025239, 0.53672097, 0.61835001]),
        'l2': np.array([-0.79501746,  0.30003095, -1.60270159,  0.26679883, -1.26162378])
    }
    
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 5,
                     transformer_series = StandardScaler(),
                     differentiation    = differentiation
                 )
    forecaster.fit(series=series_train)
    forecaster.out_sample_residuals_ = {}
    forecaster.out_sample_residuals_by_bin_ = {}
    results = forecaster._binning_out_sample_residuals(
        level="l1", y_true = y_true['l1'], y_pred = y_pred['l1']
    )

    y_true['l1'] = forecaster.transformer_series_['l1'].transform(y_true['l1'].reshape(-1, 1)).flatten()
    y_pred['l1'] = forecaster.transformer_series_['l1'].transform(y_pred['l1'].reshape(-1, 1)).flatten()
    y_true['l1'] = forecaster.differentiator_['l1'].transform(y_true['l1'])[forecaster.differentiation_max:]
    y_pred['l1'] = forecaster.differentiator_['l1'].transform(y_pred['l1'])[forecaster.differentiation_max:]
    residuals_l1 = y_true['l1'] - y_pred['l1']

    np.testing.assert_array_almost_equal(results[0], residuals_l1)
