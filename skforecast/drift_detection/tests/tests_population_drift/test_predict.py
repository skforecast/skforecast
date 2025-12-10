# Unit test predict
# ==============================================================================
from pathlib import Path
import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from ..._population_drift import PopulationDriftDetector

# Fixtures
THIS_DIR = Path(__file__).parent
data = joblib.load(THIS_DIR/'fixture_data_population_drift.joblib')
results_nannyml = joblib.load(THIS_DIR/'fixture_results_nannyml.joblib')
results_multiseries = joblib.load(THIS_DIR/'fixture_results_multiseries.joblib')
summary_multiseries = joblib.load(THIS_DIR/'fixture_summary_multiseries.joblib')


# NOTE: Code used to generate fixture_results_nannyml
# detector = nml.UnivariateDriftCalculator(
#     column_names=data_train.columns.tolist(),
#     timestamp_column_name="date_time",
#     chunk_period='M',
#     categorical_methods=['chi2', 'jensen_shannon'],
#     continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
# )
# detector.fit(reference_data=data_train.reset_index())
# results_nannyml = detector.calculate(data=data_new.reset_index())
# results_nannyml = results_nannyml.filter(period='analysis').to_df(multilevel=False)


def test_predict_NotFittedError_when_detector_not_fitted():
    """
    Test NotFittedError is raised when trying to predict before fitting the detector.
    """
    detector = PopulationDriftDetector(
        chunk_size='MS',            
        threshold=0.99,
        threshold_method='quantile'
    )
    X = data
    err_msg = (
        "This PopulationDriftDetector instance is not fitted yet. "
        "Call 'fit' with appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=err_msg):
        detector.predict(X=X)


def test_predict_ValueError_when_X_not_dataframe():
    """
    Test ValueError is raised when X is not a pandas DataFrame.
    """
    detector = PopulationDriftDetector(
        chunk_size='MS',            
        threshold=0.99,
        threshold_method='quantile'
    )
    detector.fit(data)
    X = 'not a dataframe'
    err_msg = f"`X` must be a pandas DataFrame. Got {type(X)} instead."
    with pytest.raises(ValueError, match=err_msg):
        detector.predict(X=X)


def test_predict_ValueError_when_chunk_size_is_frequency_but_X_index_not_datetimeindex():
    """
    Test ValueError is raised when chunk_size is a pandas frequency but X does 
    not have a DatetimeIndex.
    """
    detector = PopulationDriftDetector(
        chunk_size='MS',            
        threshold=0.99,
        threshold_method='quantile'
    )
    detector.fit(data)
    X = data.reset_index()
    err_msg = "`chunk_size` is a pandas frequency but `X` does not have a DatetimeIndex."
    with pytest.raises(ValueError, match=err_msg):
        detector.predict(X=X)

# TODO: Add threshold columns to the test
def test_predict_output_equivalence_nannyml():
    """
    Test that the output of PopulationDriftDetector.predict is equivalent to
    the output of NannyML's univariate drift detection for each series_id and feature.
    """
    data_train = data.iloc[: len(data) // 2].copy()
    data_new  = data.iloc[len(data) // 2 :].copy()
    data_train['weather'] = data_train['weather'].astype('category')
    data_new['weather'] = pd.Categorical(
        data_new['weather'], categories=data_train['weather'].cat.categories
    )

    detector = PopulationDriftDetector(
        chunk_size='MS',            
        threshold=3,
        threshold_method='std'
    )
    detector.fit(data_train)
    results_skforecast, _ = detector.predict(data_new)

    features = results_skforecast["feature"].unique()
    for feature in features:
        print(f"Feature: {feature}")
        cols = [
            "chunk_index",
            "chunk_start_date",
            "chunk_end_date",
            f"{feature}_kolmogorov_smirnov_value",
            f"{feature}_kolmogorov_smirnov_upper_threshold",
            f"{feature}_jensen_shannon_value",
            f"{feature}_jensen_shannon_upper_threshold",
            f"{feature}_chi2_value",
            f"{feature}_chi2_upper_threshold",
        ]
        # select columns if they exist in results_skforecast
        cols = [col for col in cols if col in results_nannyml.columns]
        df_nannyml = results_nannyml[cols].copy()
        df_nannyml = df_nannyml.rename(
            columns={
                "chunk_index": "chunk",
                "chunk_start_date": "chunk_start",
                "chunk_end_date": "chunk_end",
                f"{feature}_kolmogorov_smirnov_value": "ks_statistic",
                f"{feature}_kolmogorov_smirnov_upper_threshold": "ks_threshold",
                f"{feature}_jensen_shannon_value": "js_statistic",
                f"{feature}_jensen_shannon_upper_threshold": "js_threshold",
                f"{feature}_chi2_value": "chi2_statistic",
                f"{feature}_chi2_upper_threshold": "chi2_threshold",
            }
        )
        df_nannyml["feature"] = feature

        if "chi2_statistic" not in df_nannyml.columns:
            df_nannyml["chi2_statistic"] = np.nan
            df_nannyml["chi2_threshold"] = np.nan
            df_nannyml = df_nannyml.astype(
                {"chi2_statistic": float, "chi2_threshold": float}
            )
        if "ks_statistic" not in df_nannyml.columns:
            df_nannyml["ks_statistic"] = np.nan
            df_nannyml["ks_threshold"] = np.nan
            df_nannyml = df_nannyml.astype(
                {"ks_statistic": float, "ks_threshold": float}
            )
        if "js_statistic" not in df_nannyml.columns:
            df_nannyml["js_statistic"] = np.nan
            df_nannyml["js_threshold"] = np.nan
            df_nannyml = df_nannyml.astype(
                {"js_statistic": float, "js_threshold": float}
            )

        df_nannyml = df_nannyml.astype(
            {"chi2_statistic": float, "chi2_threshold": float,
             "ks_statistic": float, "ks_threshold": float,
             "js_statistic": float, "js_threshold": float}
        )

        df_nannyml = df_nannyml[
            [
                "chunk",
                # "chunk_start", # Different formatting in nannyml and skforecast
                # "chunk_end",   # Different formatting in nannyml and skforecast
                "feature",
                "ks_statistic",
                "ks_threshold",
                "chi2_statistic",
                # "chi2_threshold", # Nanny usses the p-value as threshold
                "js_statistic",
                "js_threshold",
            ]
        ]

        df_skforecast = results_skforecast.query(f"feature == '{feature}'")[
            [
                "chunk",
                # "chunk_start",
                # "chunk_end",
                "feature",
                "ks_statistic",
                "ks_threshold",
                "chi2_statistic",
                # "chi2_threshold", # Nanny usses the p-value as threshold
                "js_statistic",
                "js_threshold",
            ]
        ].reset_index(drop=True)

        pd.testing.assert_frame_equal(
            df_nannyml, 
            df_skforecast
        )


def test_predict_output_when_multiple_series():
    """
    Test that PopulationDriftDetector.predict works when data contains multiple series.
    """

    data_multiseries = pd.concat(
        [
            data.assign(series='series_1'),
            data.assign(series='series_2'),
            data.assign(series='series_3')
        ]
    ).set_index('series', append=True).swaplevel(0, 1)

    detector = PopulationDriftDetector(
        chunk_size="MS",
        threshold=0.95,
        threshold_method='quantile'
    )
    detector.fit(data_multiseries)
    results, summary = detector.predict(data_multiseries)

    pd.testing.assert_frame_equal(results, results_multiseries)
    pd.testing.assert_frame_equal(summary, summary_multiseries)
