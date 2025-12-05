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

#TODO: apply the following cahnges to fixture_results_multiseries
results_multiseries = results_multiseries.rename(columns={
    'statistic_ks': 'ks_statistic',
    'statistic_chi2': 'chi2_statistic',
    'jensen_shannon': 'js_statistic',
    'threshold_ks': 'ks_threshold',
    'threshold_chi2': 'chi2_threshold',
    'threshold_js': 'js_threshold'
})

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
        chunk_size='ME',            
        threshold=0.99
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
        chunk_size='ME',            
        threshold=0.99
    )
    detector.fit(data)
    X = 'not a dataframe'
    err_msg = f"`X` must be a pandas DataFrame. Got {type(X)} instead."
    with pytest.raises(ValueError, match=err_msg):
        detector.predict(X=X)


def test_predict_ValueError_when_chunk_size_is_DateOffset_but_X_index_not_datetimeindex():
    """
    Test ValueError is raised when chunk_size is a pandas DateOffset but X does 
    not have a DatetimeIndex.
    """
    detector = PopulationDriftDetector(
        chunk_size='ME',            
        threshold=0.99
    )
    detector.fit(data)
    X = data.reset_index()
    err_msg = "`chunk_size` is a pandas frequency but `X` does not have a DatetimeIndex."
    with pytest.raises(ValueError, match=err_msg):
        detector.predict(X=X)


def test_predict_output_equivalence_nannyml():
    """
    Test that the output of PopulationDriftDetector.predict is equivalent to
    the output of NannyML's univariate drift detection for each series_id and feature.
    """
    data_train = data.iloc[: len(data) // 2].copy()
    data_new  = data.iloc[len(data) // 2 :].copy()
    data_train['weather'] = data_train['weather'].astype('category')
    data_new['weather'] = pd.Categorical(data_new['weather'], categories=data_train['weather'].cat.categories)

    detector = PopulationDriftDetector(
        chunk_size='ME',            
        threshold=0.99
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
            f"{feature}_jensen_shannon_value",
            f"{feature}_chi2_value",
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
                f"{feature}_jensen_shannon_value": "js_statistic",
                f"{feature}_chi2_value": "chi2_statistic",
            }
        )
        df_nannyml["feature"] = feature
        if "chi2_statistic" not in df_nannyml.columns:
            df_nannyml["chi2_statistic"] = np.nan
        if "ks_statistic" not in df_nannyml.columns:
            df_nannyml["ks_statistic"] = np.nan
        if "js_statistic" not in df_nannyml.columns:
            df_nannyml["js_statistic"] = np.nan

        df_nannyml = df_nannyml[
            [
                "chunk",
                "chunk_start",
                "chunk_end",
                "feature",
                "ks_statistic",
                "chi2_statistic",
                "js_statistic",
            ]
        ]

        df_skforecast = results_skforecast.query(f"feature == '{feature}'")[
            [
                "chunk",
                "chunk_start",
                "chunk_end",
                "feature",
                "ks_statistic",
                "chi2_statistic",
                "js_statistic",
            ]
        ]

        df_all = pd.merge(
            df_nannyml,
            df_skforecast,
            on=["chunk", "chunk_start", "feature"],
            suffixes=("_nannyml", "_skforecast"),
        )
        pd.testing.assert_series_equal(
            df_all["ks_statistic_nannyml"],
            df_all["ks_statistic_skforecast"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            df_all["chi2_statistic_nannyml"],
            df_all["chi2_statistic_skforecast"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            df_all["js_statistic_nannyml"],
            df_all["js_statistic_skforecast"],
            check_names=False,
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
        chunk_size='ME',            
        threshold=0.95
    )
    detector.fit(data_multiseries)
    results, summary = detector.predict(data_multiseries)

    pd.testing.assert_frame_equal(results, results_multiseries)

    #TODO: update summary_multiseries fixture to add 'chunks_with_drift' column
    pd.testing.assert_frame_equal(summary.drop(columns=['chunks_with_drift']), summary_multiseries)
    