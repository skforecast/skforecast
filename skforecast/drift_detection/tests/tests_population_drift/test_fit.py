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
nannyml_fitted_stats = joblib.load(THIS_DIR/'fixture_nannyml_fitted_stats.joblib')
data = joblib.load(THIS_DIR/'fixture_data_population_drift.joblib')
data['weather'] = data['weather'].astype('category')
data_multiseries = pd.concat(
    [
        data.assign(series='series_1'),
        data.assign(series='series_2'),
        data.assign(series='series_3')
    ]
).set_index('series', append=True).swaplevel(0, 1)


def test_fit_exception_X_is_not_dataframe():
    """
    Test that fit raises an exception when X is not a pandas DataFrame.
    """
    detector = PopulationDriftDetector(chunk_size="ME", threshold=0.95)
    X = "not a dataframe"
    err_msg = f"`X` must be a pandas DataFrame. Got {type(X)} instead."
    with pytest.raises(ValueError, match=err_msg):
        detector.fit(X=X)


def test_fit_exception_chunk_size_is_pandas_frequency_str_but_X_has_no_datetime_index():
    """
    Test that fit raises an exception when chunk_size is a string compatible with
    pandas frequency but X has no datetime index.
    """
    detector = PopulationDriftDetector(chunk_size="ME", threshold=0.95)
    X = pd.DataFrame({"A": np.random.rand(100), "B": np.random.rand(100)})
    err_msg = (
        "`chunk_size` is a pandas frequency but `X` does not have a DatetimeIndex."
    )
    with pytest.raises(ValueError, match=err_msg):
        detector.fit(X=X)


def test_fit_exception_warning_when_feature_is_all_nan():
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

    detector = PopulationDriftDetector(
        chunk_size="MS",
        threshold=0.95,
        threshold_method='quantile'
    )
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
        "is_fitted": True,
        "empirical_threshold_ks_": {
            "holiday": np.float64(0.038164011637971604),
            "workingday": np.float64(0.06157834101382487),
            "weather": np.float64(np.nan),
            "temp": np.float64(0.6938360215053764),
            "atemp": np.float64(0.6703988095238095),
            "hum": np.float64(0.25918251703915024),
            "windspeed": np.float64(0.1879290497701695),
            "users": np.float64(0.3624611154630254),
            "month": np.float64(0.9557142857142857),
            "hour": np.float64(0.0),
            "weekday": np.float64(0.057942396313364064),
        },
        "empirical_threshold_chi2_": {
            "holiday": np.float64(np.nan),
            "workingday": np.float64(np.nan),
            "weather": np.float64(138.6260973458945),
            "temp": np.float64(np.nan),
            "atemp": np.float64(np.nan),
            "hum": np.float64(np.nan),
            "windspeed": np.float64(np.nan),
            "users": np.float64(np.nan),
            "month": np.float64(np.nan),
            "hour": np.float64(np.nan),
            "weekday": np.float64(np.nan),
        },
        "empirical_threshold_js_": {
            "holiday": np.float64(0.12534980026689746),
            "workingday": np.float64(0.057738428555656876),
            "weather": np.float64(0.18473248925466101),
            "temp": np.float64(0.7107152127197462),
            "atemp": np.float64(0.7065948215181463),
            "hum": np.float64(0.30088895106341096),
            "windspeed": np.float64(0.19564880494054016),
            "users": np.float64(0.41486025162255186),
            "month": np.float64(0.9350048038295908),
            "hour": np.float64(0.0),
            "weekday": np.float64(0.05085543161075281),
        },
        "ref_ranges_": {
            "holiday": (np.float64(0.0), np.float64(1.0)),
            "workingday": (np.float64(0.0), np.float64(1.0)),
            "temp": (np.float64(0.8200000000000001), np.float64(41.0)),
            "atemp": (np.float64(0.0), np.float64(50.0)),
            "hum": (np.float64(0.0), np.float64(100.0)),
            "windspeed": (np.float64(0.0), np.float64(56.9969)),
            "users": (np.float64(1.0), np.float64(977.0)),
            "month": (np.int64(1), np.int64(12)),
            "hour": (np.int64(0), np.int64(23)),
            "weekday": (np.int64(0), np.int64(6)),
        },
        "ref_categories_": {"weather": ["clear", "mist", "rain"]},
        "n_chunks_reference_data_": 24,
        "detectors_": {},
        "threshold": 0.95,
        "chunk_size": "MS",
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

    detector = PopulationDriftDetector(
        chunk_size="MS",
        threshold=0.95,
        threshold_method='quantile'
    )
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
        "is_fitted": True,
        "empirical_threshold_ks_": {
            "holiday": np.float64(0.038164011637971604),
            "workingday": np.float64(0.06157834101382487),
            "weather": np.float64(np.nan),
            "temp": np.float64(0.6938360215053764),
            "atemp": np.float64(0.6703988095238095),
            "hum": np.float64(0.25918251703915024),
            "windspeed": np.float64(0.1879290497701695),
            "users": np.float64(0.3624611154630254),
            "month": np.float64(0.9557142857142857),
            "hour": np.float64(0.0),
            "weekday": np.float64(0.057942396313364064),
        },
        "empirical_threshold_chi2_": {
            "holiday": np.float64(np.nan),
            "workingday": np.float64(np.nan),
            "weather": np.float64(138.6260973458945),
            "temp": np.float64(np.nan),
            "atemp": np.float64(np.nan),
            "hum": np.float64(np.nan),
            "windspeed": np.float64(np.nan),
            "users": np.float64(np.nan),
            "month": np.float64(np.nan),
            "hour": np.float64(np.nan),
            "weekday": np.float64(np.nan),
        },
        "empirical_threshold_js_": {
            "holiday": np.float64(0.12534980026689746),
            "workingday": np.float64(0.057738428555656876),
            "weather": np.float64(0.18473248925466101),
            "temp": np.float64(0.7107152127197462),
            "atemp": np.float64(0.7065948215181463),
            "hum": np.float64(0.30088895106341096),
            "windspeed": np.float64(0.19564880494054016),
            "users": np.float64(0.41486025162255186),
            "month": np.float64(0.9350048038295908),
            "hour": np.float64(0.0),
            "weekday": np.float64(0.05085543161075281),
        },
        "ref_ranges_": {
            "holiday": (np.float64(0.0), np.float64(1.0)),
            "workingday": (np.float64(0.0), np.float64(1.0)),
            "temp": (np.float64(0.8200000000000001), np.float64(41.0)),
            "atemp": (np.float64(0.0), np.float64(50.0)),
            "hum": (np.float64(0.0), np.float64(100.0)),
            "windspeed": (np.float64(0.0), np.float64(56.9969)),
            "users": (np.float64(1.0), np.float64(977.0)),
            "month": (np.int64(1), np.int64(12)),
            "hour": (np.int64(0), np.int64(23)),
            "weekday": (np.int64(0), np.int64(6)),
        },
        "ref_categories_": {"weather": ["clear", "mist", "rain"]},
        "n_chunks_reference_data_": 24,
        "detectors_": {},
        "threshold": 0.95,
        "chunk_size": "MS",
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


def test_empirical_distributions_match_nannyml():
    """
    Test that the empirical distributions computed by PopulationDriftDetector
    during fit match the ones from NannyML implementation.

    Warning: skforecast only matches NannyML when the threshold_method='std' is used.
    
    This test verifies that the Jensen-Shannon, Kolmogorov-Smirnov, and Chi2
    statistics calculated during the fit phase match the reference values
    computed by NannyML with the same configuration (chunk_size='MS', 
    categorical_methods=['chi2', 'jensen_shannon'], 
    continuous_methods=['kolmogorov_smirnov', 'jensen_shannon']).
    
    The reference values were generated using NannyML 0.13.1 and saved in
    'fixture_nannyml_fitted_stats.joblib'.
    """
    # Fit the detector
    detector_sk = PopulationDriftDetector(
        chunk_size='MS',
        threshold=3,
        threshold_method='std'
    )
    detector_sk.fit(data)
    
    # Compare Jensen-Shannon statistics for all features
    for feature in data.columns:
        skforecast_values = np.array(detector_sk.empirical_dist_js_[feature])
        nannyml_values = nannyml_fitted_stats['empirical_dist_js_'][feature]
        np.testing.assert_array_almost_equal(
            skforecast_values, 
            nannyml_values, 
            decimal=5,
            err_msg=f"Jensen-Shannon statistics mismatch for feature '{feature}'"
        )
    
    # Compare Kolmogorov-Smirnov statistics for numerical features
    for feature in data.select_dtypes(include=['number']).columns:
        skforecast_values = np.array(detector_sk.empirical_dist_ks_[feature])
        nannyml_values = nannyml_fitted_stats['empirical_dist_ks_'][feature]
        np.testing.assert_array_almost_equal(
            skforecast_values, 
            nannyml_values, 
            decimal=5,
            err_msg=f"Kolmogorov-Smirnov statistics mismatch for feature '{feature}'"
        )
    
    # Compare Chi2 statistics for categorical features
    for feature in data.select_dtypes(include=['category', 'object']).columns:
        skforecast_values = np.array(detector_sk.empirical_dist_chi2_[feature])
        nannyml_values = nannyml_fitted_stats['empirical_dist_chi2_'][feature]
        np.testing.assert_array_almost_equal(
            skforecast_values, 
            nannyml_values, 
            decimal=5,
            err_msg=f"Chi2 statistics mismatch for feature '{feature}'"
        )