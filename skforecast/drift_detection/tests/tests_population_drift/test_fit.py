# Unit test fit
# ==============================================================================
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import pytest
from ....drift_detection import PopulationDriftDetector


# fixtures
THIS_DIR = Path(__file__).parent
data = joblib.load(THIS_DIR/'fixture_data_population_drift.joblib')
data_multiseries = pd.concat(
    [
        data.assign(series='series_1'),
        data.assign(series='series_2'),
        data.assign(series='series_3')
    ]
).set_index('series', append=True).swaplevel(0,1)


def test_fit_exception_X_is_not_dataframe():
    """
    Test that fit raises an exception when X is not a pandas DataFrame.
    """
    detector = PopulationDriftDetector(chunk_size="ME", threshold=0.95)
    X = "not a dataframe"
    err_msg = f"`X` must be a pandas DataFrame. Got {type(X)} instead."
    with pytest.raises(ValueError, match=err_msg):
        detector.fit(X=X)


def test_fit_exception_chunk_size_is_pandas_DateOffset_str_but_X_has_no_datetime_index():
    """
    Test that fit raises an exception when chunk_size is a string compatible with
    pandas DateOffset but X has no datetime index.
    """
    detector = PopulationDriftDetector(chunk_size="ME", threshold=0.95)
    X = pd.DataFrame({"A": np.random.rand(100), "B": np.random.rand(100)})
    err_msg = (
        "`chunk_size` is a pandas DateOffset but `X` does not have a DatetimeIndex."
    )
    with pytest.raises(ValueError, match=err_msg):
        detector.fit(X=X)


def test_fit_exception_warning_when_fature_is_all_nan():
    """
    Test that fit raises a warning when a feature is all NaN.
    """
    detector = PopulationDriftDetector(chunk_size="ME", threshold=0.95)
    X = data.copy()
    X.loc[:, "temp"] = np.nan
    warn_msg = (
        "Feature 'temp' contains only NaN values in the reference dataset. "
        "Drift detection skipped."
    )
    with pytest.warns(UserWarning, match=warn_msg):
        detector.fit(X=X)


def test_fit_stored_attributes():
    """
    Test that the attributes of PopulationDriftDetector are correctly stored after calling fit.
    """

    detector = PopulationDriftDetector(chunk_size="ME", threshold=0.95)
    detector.fit(data)

    expected_results = {
        "ref_features_": [
            "holiday",
            "workingday",
            "weather",
            "temp",
            "atemp",
            "hum",
            "windspeed",
            "users",
            "month",
            "hour",
            "weekday",
        ],
        "is_fitted_": True,
        "empirical_threshold_ks_": {
            "holiday": 0.03659032405160117,
            "workingday": 0.05896694761925775,
            "weather": np.nan,
            "temp": 0.6644120588970772,
            "atemp": 0.6419687642498859,
            "hum": 0.24854575163398676,
            "windspeed": 0.18021650326797378,
            "users": 0.3483888008504226,
            "month": 0.9151846785225718,
            "hour": 0.0,
            "weekday": 0.055485194828118825,
        },
        "empirical_threshold_chi2_": {
            "holiday": np.nan,
            "workingday": np.nan,
            "weather": 128.71740772446282,
            "temp": np.nan,
            "atemp": np.nan,
            "hum": np.nan,
            "windspeed": np.nan,
            "users": np.nan,
            "month": np.nan,
            "hour": np.nan,
            "weekday": np.nan,
        },
        "empirical_threshold_js_": {
            "holiday": 0.12263338812190272,
            "workingday": 0.05536149547827185,
            "weather": 0.14695782935496615,
            "temp": 0.6845879616136854,
            "atemp": 0.6797815908883446,
            "hum": 0.28687940327125355,
            "windspeed": 0.1881001299842457,
            "users": 0.4033478565343228,
            "month": 0.892720109868587,
            "hour": 0.0,
            "weekday": 0.04868637565402696,
        },
        "ref_ranges_": {
            "holiday": (0.0, 1.0),
            "workingday": (0.0, 1.0),
            "temp": (0.8200000000000001, 41.0),
            "atemp": (0.0, 50.0),
            "hum": (0.0, 100.0),
            "windspeed": (0.0, 56.9969),
            "users": (1.0, 977.0),
            "month": (1, 12),
            "hour": (0, 23),
            "weekday": (0, 6),
        },
        "ref_categories_": {"weather": ["clear", "mist", "rain"]},
        "n_chunks_reference_data_": 24,
        "detectors_": {},
        "threshold": 0.95,
        "chunk_size": "ME",
    }

    for attr, expected_value in expected_results.items():
        value = getattr(detector, attr)
        if isinstance(value, dict):
            for key in expected_value.keys():
                if isinstance(expected_value[key], float) and np.isnan(expected_value[key]):
                    assert np.isnan(value[key])
                elif isinstance(expected_value[key], list):
                    assert value[key] == expected_value[key]
                else:
                    np.testing.assert_almost_equal(value[key], expected_value[key], decimal=5)
        elif isinstance(value, list):
            assert value == expected_value
        elif isinstance(value, float):
            np.testing.assert_almost_equal(value, expected_value, decimal=5)
        else:
            assert value == expected_value


def test_fit_stored_attributes_multiseries():
    """
    Test that the attributes of PopulationDriftDetector are correctly stored after calling fit
    when data contains multiple series.
    """

    detector = PopulationDriftDetector(chunk_size="ME", threshold=0.95)
    detector.fit(data_multiseries)

    expected_results = {
        "ref_features_": [
            "holiday",
            "workingday",
            "weather",
            "temp",
            "atemp",
            "hum",
            "windspeed",
            "users",
            "month",
            "hour",
            "weekday",
        ],
        "is_fitted_": True,
        "empirical_threshold_ks_": {
            "holiday": 0.03659032405160117,
            "workingday": 0.05896694761925775,
            "weather": np.nan,
            "temp": 0.6644120588970772,
            "atemp": 0.6419687642498859,
            "hum": 0.24854575163398676,
            "windspeed": 0.18021650326797378,
            "users": 0.3483888008504226,
            "month": 0.9151846785225718,
            "hour": 0.0,
            "weekday": 0.055485194828118825,
        },
        "empirical_threshold_chi2_": {
            "holiday": np.nan,
            "workingday": np.nan,
            "weather": 128.71740772446282,
            "temp": np.nan,
            "atemp": np.nan,
            "hum": np.nan,
            "windspeed": np.nan,
            "users": np.nan,
            "month": np.nan,
            "hour": np.nan,
            "weekday": np.nan,
        },
        "empirical_threshold_js_": {
            "holiday": 0.12263338812190272,
            "workingday": 0.05536149547827185,
            "weather": 0.14695782935496615,
            "temp": 0.6845879616136854,
            "atemp": 0.6797815908883446,
            "hum": 0.28687940327125355,
            "windspeed": 0.1881001299842457,
            "users": 0.4033478565343228,
            "month": 0.892720109868587,
            "hour": 0.0,
            "weekday": 0.04868637565402696,
        },
        "ref_ranges_": {
            "holiday": (0.0, 1.0),
            "workingday": (0.0, 1.0),
            "temp": (0.8200000000000001, 41.0),
            "atemp": (0.0, 50.0),
            "hum": (0.0, 100.0),
            "windspeed": (0.0, 56.9969),
            "users": (1.0, 977.0),
            "month": (1, 12),
            "hour": (0, 23),
            "weekday": (0, 6),
        },
        "ref_categories_": {"weather": ["clear", "mist", "rain"]},
        "n_chunks_reference_data_": 24,
        "detectors_": {},
        "threshold": 0.95,
        "chunk_size": "ME",
    }

    for series_id in ['series_1', 'series_2', 'series_3']:
        detector_series = detector.detectors_[series_id]
        for attr, expected_value in expected_results.items():
            value = getattr(detector_series, attr)
            if isinstance(value, dict):
                for key in expected_value.keys():
                    if isinstance(expected_value[key], float) and np.isnan(expected_value[key]):
                        assert np.isnan(value[key])
                    elif isinstance(expected_value[key], list):
                        assert value[key] == expected_value[key]
                    else:
                        np.testing.assert_almost_equal(value[key], expected_value[key], decimal=5)
            elif isinstance(value, list):
                assert value == expected_value
            elif isinstance(value, float):
                np.testing.assert_almost_equal(value, expected_value, decimal=5)
            else:
                assert value == expected_value
