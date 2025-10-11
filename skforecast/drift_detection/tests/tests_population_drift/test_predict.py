# Unit test predict
# ==============================================================================
import re
import pytest
import pandas as pd
import numpy as np
import nannyml as nml
import warnings
from skforecast.drift_detection import PopulationDriftDetector
import joblib

#fixtures
data = joblib.load('./fixture_data_population_drift.joblib')


def test_predict_output_equivalence_nannyml():
    """
    Test that the output of PopulationDriftDetector.predict is equivalent to
    the output of NannyML's univariate drift detection for each series_id and feature.
    """
    data_train = data.iloc[: len(data)//2].copy()
    data_new  = data.iloc[len(data)//2 :].copy()
    data_train['weather'] = data_train['weather'].astype('category')
    data_new['weather'] = pd.Categorical(data_new['weather'], categories=data_train['weather'].cat.categories)
    
    detector = nml.UnivariateDriftCalculator(
        column_names=data_train.columns.tolist(),
        timestamp_column_name="date_time",
        chunk_period='M',
        categorical_methods=['chi2', 'jensen_shannon'],
        continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
    )
    detector.fit(reference_data=data_train.reset_index())
    results_nannyml = detector.calculate(data=data_new.reset_index())
    results_nannyml = results_nannyml.filter(period='analysis').to_df(multilevel=False)

    detector = PopulationDriftDetector(
        chunk_size='ME',            
        threshold=0.99
    )
    detector.fit(data_train)
    results_skforecast = detector.predict(data_new)

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
                f"{feature}_jensen_shannon_value": "jensen_shannon",
                f"{feature}_chi2_value": "chi2_statistic",
            }
        )
        df_nannyml["feature"] = feature
        if "chi2_statistic" not in df_nannyml.columns:
            df_nannyml["chi2_statistic"] = np.nan
        if "ks_statistic" not in df_nannyml.columns:
            df_nannyml["ks_statistic"] = np.nan
        if "jensen_shannon" not in df_nannyml.columns:
            df_nannyml["jensen_shannon"] = np.nan

        df_nannyml = df_nannyml[
            [
                "chunk",
                "chunk_start",
                "chunk_end",
                "feature",
                "ks_statistic",
                "chi2_statistic",
                "jensen_shannon",
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
                "jensen_shannon",
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
            df_all["jensen_shannon_nannyml"],
            df_all["jensen_shannon_skforecast"],
            check_names=False,
        )
