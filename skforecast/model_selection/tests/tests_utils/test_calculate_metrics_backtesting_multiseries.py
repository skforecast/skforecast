# Unit test calculate_metrics_multiseries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from skforecast.metrics import mean_absolute_scaled_error, add_y_train_argument
from skforecast.model_selection._utils import _calculate_metrics_backtesting_multiseries

# Fixtures
data = pd.DataFrame(
    data={
        "item_1": [
            8.253175, 22.777826, 27.549099, 25.895533, 21.379238, 21.106643,
            20.533871, 20.069327, 20.006161, 21.620184, 21.717691, 21.751748,
            21.758617, 20.784194, 18.976196, 20.228468, 26.636444, 29.245869,
            24.772249, 24.018768, 22.503533, 20.794986, 23.981037, 28.018830,
            28.747482, 23.908368, 21.423930, 24.786455, 24.615778, 27.388275,
            25.724191, 22.825491, 23.066582, 23.788066, 23.360304, 23.119966,
            21.763739, 23.008517, 22.861086, 22.807790, 23.424717, 22.208947,
            19.558775, 20.788390, 23.619240, 25.061150, 27.646380, 25.609772,
            22.504042, 20.838095
        ],
        "item_2": [
            21.047727, 26.578125, 31.751042, 24.567708, 18.191667, 17.812500,
            19.510417, 24.098958, 20.223958, 19.161458, 16.042708, 14.815625,
            17.031250, 17.009375, 17.096875, 19.255208, 28.060417, 28.779167,
            19.265625, 19.178125, 19.688542, 21.690625, 25.332292, 26.675000,
            26.611458, 19.759375, 20.038542, 24.680208, 25.032292, 28.111458,
            21.542708, 16.605208, 18.593750, 20.667708, 21.977083, 29.040625,
            18.979167, 18.459375, 17.295833, 17.282292, 20.844792, 19.858333,
            18.446875, 19.239583, 19.903125, 22.970833, 28.195833, 20.221875,
            19.176042, 21.991667
        ],
        "item_3": [
            19.429739, 28.009863, 32.078922, 27.252276, 20.357737, 19.879148,
            18.043499, 26.287368, 16.315997, 21.772584, 18.729748, 12.552534,
            18.996209, 18.534327, 15.418361, 16.304852, 30.076258, 28.886334,
            20.286651, 21.367727, 20.248170, 19.799975, 25.931558, 27.698196,
            30.725005, 19.573577, 23.310162, 24.959233, 24.399246, 29.094136,
            22.639513, 18.372362, 21.256450, 22.430527, 19.575067, 31.767626,
            20.086271, 21.380186, 17.553807, 17.369879, 21.829746, 16.208510,
            25.067215, 21.863615, 17.887458, 23.005424, 25.013939, 22.142083,
            23.673005, 25.238480
        ],
    },
    index=pd.date_range(start="2012-01-01", end="2012-02-19"),
)

predictions = pd.DataFrame(
    data={
        "item_1": [
            25.849411, 24.507137, 23.885447, 23.597504, 23.464140, 23.402371,
            23.373762, 23.360511, 23.354374, 23.351532, 23.354278, 23.351487,
            23.350195, 23.349596, 23.349319, 23.349190, 23.349131, 23.349103,
            23.349090, 23.349084, 23.474207, 23.407034, 23.375922, 23.361512,
            23.354837
        ],
        "item_2": [
            24.561460, 23.611980, 23.172218, 22.968536, 22.874199, 22.830506,
            22.810269, 22.800896, 22.796555, 22.794544, 22.414996, 22.617821,
            22.711761, 22.755271, 22.775423, 22.784756, 22.789079, 22.791082,
            22.792009, 22.792439, 21.454419, 22.172918, 22.505700, 22.659831,
            22.731219
        ],
        "item_3": [
            26.168069, 24.057472, 23.079925, 22.627163, 22.417461, 22.320335,
            22.275350, 22.254515, 22.244865, 22.240395, 21.003848, 21.665604,
            21.972104, 22.114063, 22.179813, 22.210266, 22.224370, 22.230903,
            22.233929, 22.235330, 20.222212, 21.303581, 21.804429, 22.036402,
            22.143843
        ],
    },
    index=pd.date_range(start="2012-01-26", periods=25)
)
predictions = (
    predictions.melt(var_name="level", value_name="pred", ignore_index=False)
    .reset_index()
    .sort_values(by=["index", "level"])
    .set_index("index")
    .rename_axis('idx', axis=0)
    .set_index('level', append=True)
)
predictions_missing_level = predictions.query("level != 'item_3'").copy()

predictions_different_length = pd.DataFrame(
    data={
        "item_1": [
            25.849411, 24.507137, 23.885447, 23.597504, 23.464140, 23.402371,
            23.373762, 23.360511, 23.354374, 23.351532, 23.354278, 23.351487,
            23.350195, 23.349596, 23.349319, 23.349190, 23.349131, 23.349103,
            23.349090, 23.349084, 23.474207, 23.407034, 23.375922, 23.361512,
            23.354837
        ],
        "item_2": [
            24.561460, 23.611980, 23.172218, 22.968536, 22.874199, 22.830506,
            22.810269, 22.800896, 22.796555, 22.794544, 22.414996, 22.617821,
            22.711761, 22.755271, 22.775423, 22.784756, 22.789079, 22.791082,
            22.792009, 22.792439, 21.454419, 22.172918, 22.505700, 22.659831,
            22.731219
        ],
        "item_3": [
            26.168069, 24.057472, 23.079925, 22.627163, 22.417461, 22.320335,
            22.275350, 22.254515, 22.244865, 22.240395, 21.003848, 21.665604,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, np.nan
        ],
    },
    index=pd.date_range(start="2012-01-26", periods=25)
)
predictions_different_length = (
    predictions_different_length.melt(var_name="level", value_name="pred", ignore_index=False)
    .reset_index()
    .sort_values(by=["index", "level"])
    .set_index("index")
    .rename_axis('idx', axis=0)
    .set_index('level', append=True)
)

span_index = pd.date_range(start="2012-01-01", end="2012-02-19", freq="D")

folds = [
    [0, [0, 25], [24, 25], [25, 35], [25, 35], True],
    [1, [0, 25], [34, 35], [35, 45], [35, 45], False],
    [2, [0, 25], [44, 45], [45, 50], [45, 50], False],
]
window_size = 2
levels = ["item_1", "item_2", "item_3"]


def custom_metric(y_true, y_pred):  # pragma: no cover
    """
    Calculate the mean absolute error excluding predictions between '2012-01-05'
    and '2012-01-10'.
    """
    mask = (y_true.index < '2012-01-05') | (y_true.index > '2012-01-10')
    metric = mean_absolute_error(y_true[mask], y_pred[mask])
    
    return metric


# NOTE: Ignored, inputs checks are commented out in the function.
# def test_calculate_metrics_backtesting_multiseries_input_types():
#     """
#     Check if function raises errors when input parameters have wrong types.
#     """

#     # Mock inputs
#     series_df = pd.DataFrame(
#         {"time": pd.date_range(start="2020-01-01", periods=3), "value": [1, 2, 3]}
#     )
#     predictions = pd.DataFrame(
#         {
#             "time": pd.date_range(start="2020-01-01", periods=3),
#             "predicted": [1.5, 2.5, 3.5],
#         }
#     )
#     folds = [{"train": (0, 1), "test": (2, 3)}]
#     span_index = pd.date_range(start="2020-01-01", periods=3)
#     window_size = 2
#     metrics = ["mean_absolute_error"]
#     levels = ["level1", "level2"]

#     # Test invalid type for series
#     msg = re.escape(
#         "`series` must be a pandas DataFrame or a dictionary of pandas DataFrames."
#     )
#     with pytest.raises(TypeError, match=msg):
#         _calculate_metrics_backtesting_multiseries(
#             "invalid_series_type", predictions, folds, span_index, window_size, metrics, levels
#         )

#     # Test invalid type for predictions
#     msg = re.escape("`predictions` must be a pandas DataFrame.")
#     with pytest.raises(TypeError, match=msg):
#         _calculate_metrics_backtesting_multiseries(
#             series_df, "invalid_predictions_type", folds, span_index, window_size, metrics, levels
#         )

#     # Test invalid type for folds
#     msg = re.escape("`folds` must be a list.")
#     with pytest.raises(TypeError, match=msg):
#         _calculate_metrics_backtesting_multiseries(
#             series_df, predictions, "invalid_folds_type", span_index, window_size, metrics, levels
#         )

#     # Test invalid type for span_index
#     msg = re.escape("`span_index` must be a pandas DatetimeIndex or pandas RangeIndex.")
#     with pytest.raises(TypeError, match=msg):
#         _calculate_metrics_backtesting_multiseries(
#             series_df, predictions, folds, "invalid_span_index_type", window_size, metrics, levels
#         )

#     # Test invalid type for window_size
#     msg = re.escape("`window_size` must be an integer.")
#     with pytest.raises(TypeError, match=msg):
#         _calculate_metrics_backtesting_multiseries(
#             series_df, predictions, folds, span_index, "invalid_window_size_type", metrics, levels
#         )

#     # Test invalid type for metrics
#     msg = re.escape("`metrics` must be a list.")
#     with pytest.raises(TypeError, match=msg):
#         _calculate_metrics_backtesting_multiseries(
#             series_df, predictions, folds, span_index, window_size, "invalid_metrics_type", levels
#         )

#     # Test invalid type for levels
#     msg = re.escape("`levels` must be a list.")
#     with pytest.raises(TypeError, match=msg):
#         _calculate_metrics_backtesting_multiseries(
#             series_df, predictions, folds, span_index, window_size, metrics, "invalid_levels_type"
#         )

#     # Test invalid type for add_aggregated_metric
#     msg = re.escape("`add_aggregated_metric` must be a boolean.")
#     with pytest.raises(TypeError, match=msg):
#         _calculate_metrics_backtesting_multiseries(
#             series_df,
#             predictions,
#             folds,
#             span_index,
#             window_size,
#             metrics,
#             levels,
#             add_aggregated_metric="invalid_type",
#         )


def test_calculate_metrics_backtesting_multiseries_output_when_no_aggregated_metric(
    metrics=[mean_absolute_error, mean_absolute_scaled_error]
):
    """
    Test output of _calculate_metrics_backtesting_multiseries when add_aggregated_metric=False
    """

    metrics = [add_y_train_argument(metric) for metric in metrics]
    results = _calculate_metrics_backtesting_multiseries(
        series=data,
        predictions=predictions,
        folds=folds,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=levels,
        add_aggregated_metric=False,
    )

    expected = pd.DataFrame(
        data={
            "levels": ["item_1", "item_2", "item_3"],
            "mean_absolute_error": [1.477567, 3.480129, 2.942386],
            "mean_absolute_scaled_error": [0.8388579569071319, 1.261808218733781, 0.6816085701001846],
        }
    )

    pd.testing.assert_frame_equal(results, expected)


def test_calculate_metrics_backtesting_multiseries_output_when_aggregated_metric(
    metrics=[mean_absolute_error, mean_absolute_scaled_error]
):
    """
    Test output of _calculate_metrics_backtesting_multiseries when add_aggregated_metric=True
    """

    metrics = [add_y_train_argument(metric) for metric in metrics]
    results = _calculate_metrics_backtesting_multiseries(
        series=data,
        predictions=predictions,
        folds=folds,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=levels,
        add_aggregated_metric=True,
    )

    expected = pd.DataFrame(
        data={
            "levels": [
                "item_1",
                "item_2",
                "item_3",
                "average",
                "weighted_average",
                "pooling",
            ],
            "mean_absolute_error": [
                1.477567,
                3.480129,
                2.942386,
                2.633361,
                2.633361,
                2.633361,
            ],
            "mean_absolute_scaled_error": [
                0.8388579569071319,
                1.261808218733781,
                0.6816085701001846,
                0.9274249152470325,
                0.9274249152470325,
                0.8940507968656655,
            ],
        }
    )

    pd.testing.assert_frame_equal(results, expected)


def test_calculate_metrics_backtesting_multiseries_output_when_aggregated_metric_and_customer_metric(
    metrics=[custom_metric],
):
    """
    Test output of _calculate_metrics_backtesting_multiseries when add_aggregated_metric=True
    """

    metrics = [add_y_train_argument(metric) for metric in metrics]
    results = _calculate_metrics_backtesting_multiseries(
        series=data,
        predictions=predictions,
        folds=folds,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=levels,
        add_aggregated_metric=True,
    )

    expected = pd.DataFrame(
        {
            "levels": {
                0: "item_1",
                1: "item_2",
                2: "item_3",
                3: "average",
                4: "weighted_average",
                5: "pooling",
            },
            "custom_metric": {
                0: 1.47756696,
                1: 3.48012924,
                2: 2.9423860000000004,
                3: 2.6333607333333333,
                4: 2.6333607333333333,
                5: 2.6333607333333338,
            },
        }
    )

    pd.testing.assert_frame_equal(results, expected)


def test_calculate_metrics_backtesting_multiseries_output_when_aggregated_metric_and_predictions_have_different_length(
    metrics=[mean_absolute_error, mean_absolute_scaled_error]
):
    """
    """
    metrics = [add_y_train_argument(metric) for metric in metrics]
    results = _calculate_metrics_backtesting_multiseries(
        series=data,
        predictions=predictions_different_length,
        folds=folds,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=levels,
        add_aggregated_metric=True,
    )

    expected = pd.DataFrame(
        data={
            "levels": [
                "item_1",
                "item_2",
                "item_3",
                "average",
                "weighted_average",
                "pooling",
            ],
            "mean_absolute_error": [
                1.477567,
                3.480129,
                3.173683,
                2.710460,
                2.613332,
                2.613332,
            ],
            "mean_absolute_scaled_error": [
                0.8388579569071319,
                1.261808218733781,
                0.7351889788709302,
                0.9452850515039476,
                0.9893374538302255,
                0.8872509680588639,
            ],
        }
    )

    pd.testing.assert_frame_equal(results, expected)


def test_calculate_metrics_backtesting_multiseries_output_when_aggregated_metric_and_one_level_is_not_predicted(
    metrics=[mean_absolute_error, mean_absolute_scaled_error]
):
    """
    
    """
    metrics = [add_y_train_argument(metric) for metric in metrics]
    results = _calculate_metrics_backtesting_multiseries(
        series=data,
        predictions=predictions_missing_level,
        folds=folds,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=levels,
        add_aggregated_metric=True,
    )

    expected = pd.DataFrame(
        {
            "levels": [
                "item_1",
                "item_2",
                "item_3",
                "average",
                "weighted_average",
                "pooling",
            ],
            "mean_absolute_error": [
                1.47756696,
                3.48012924,
                np.nan,
                2.4788481,
                2.4788481,
                2.4788481,
            ],
            "mean_absolute_scaled_error": [
                0.8388579569071319,
                1.261808218733781,
                np.nan,
                1.0503330878204564,
                1.0503330878204564,
                1.096968360536767,
            ],
        }
    )

    pd.testing.assert_frame_equal(results, expected)


def test_calculate_metrics_backtesting_multiseries_with_single_level():
    """
    Test that _calculate_metrics_backtesting_multiseries works correctly when 
    only one level is used (add_aggregated_metric should be False automatically).
    """
    single_level_data = data[["item_1"]].copy()
    single_level_predictions = predictions.query("level == 'item_1'").copy()
    single_level_levels = ["item_1"]
    
    metrics = [add_y_train_argument(mean_absolute_error)]
    
    results = _calculate_metrics_backtesting_multiseries(
        series=single_level_data,
        predictions=single_level_predictions,
        folds=folds,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=single_level_levels,
        add_aggregated_metric=True,  # Should be ignored when only 1 level
    )

    expected = pd.DataFrame(
        data={
            "levels": ["item_1"],
            "mean_absolute_error": [1.477567],
        }
    )

    pd.testing.assert_frame_equal(results, expected)


def test_calculate_metrics_backtesting_multiseries_with_rangeindex():
    """
    Test that _calculate_metrics_backtesting_multiseries works correctly when 
    series has RangeIndex instead of DatetimeIndex.
    """
    # Create data with RangeIndex
    data_range = data.copy()
    data_range.index = pd.RangeIndex(start=0, stop=len(data_range), step=1)
    
    predictions_range = pd.DataFrame(
        data={
            "item_1": [
                25.849411, 24.507137, 23.885447, 23.597504, 23.464140, 23.402371,
                23.373762, 23.360511, 23.354374, 23.351532, 23.354278, 23.351487,
                23.350195, 23.349596, 23.349319, 23.349190, 23.349131, 23.349103,
                23.349090, 23.349084, 23.474207, 23.407034, 23.375922, 23.361512,
                23.354837
            ],
            "item_2": [
                24.561460, 23.611980, 23.172218, 22.968536, 22.874199, 22.830506,
                22.810269, 22.800896, 22.796555, 22.794544, 22.414996, 22.617821,
                22.711761, 22.755271, 22.775423, 22.784756, 22.789079, 22.791082,
                22.792009, 22.792439, 21.454419, 22.172918, 22.505700, 22.659831,
                22.731219
            ],
            "item_3": [
                26.168069, 24.057472, 23.079925, 22.627163, 22.417461, 22.320335,
                22.275350, 22.254515, 22.244865, 22.240395, 21.003848, 21.665604,
                21.972104, 22.114063, 22.179813, 22.210266, 22.224370, 22.230903,
                22.233929, 22.235330, 20.222212, 21.303581, 21.804429, 22.036402,
                22.143843
            ],
        },
        index=pd.RangeIndex(start=25, stop=50, step=1)
    )
    predictions_range = (
        predictions_range.melt(var_name="level", value_name="pred", ignore_index=False)
        .reset_index()
        .sort_values(by=["index", "level"])
        .set_index("index")
        .rename_axis('idx', axis=0)
        .set_index('level', append=True)
    )
    
    span_index_range = pd.RangeIndex(start=0, stop=50, step=1)
    
    metrics = [add_y_train_argument(mean_absolute_error)]
    
    results = _calculate_metrics_backtesting_multiseries(
        series=data_range,
        predictions=predictions_range,
        folds=folds,
        span_index=span_index_range,
        window_size=window_size,
        metrics=metrics,
        levels=levels,
        add_aggregated_metric=False,
    )

    expected = pd.DataFrame(
        data={
            "levels": ["item_1", "item_2", "item_3"],
            "mean_absolute_error": [1.477567, 3.480129, 2.942386],
        }
    )

    pd.testing.assert_frame_equal(results, expected)


def test_calculate_metrics_backtesting_multiseries_with_multiple_folds_and_refit():
    """
    Test that _calculate_metrics_backtesting_multiseries correctly handles 
    multiple folds with different refit patterns (True/False in the last element).
    """
    # Folds with different refit patterns
    folds_mixed_refit = [
        [0, [0, 25], [24, 25], [25, 35], [25, 35], True],   # refit
        [1, [0, 30], [29, 30], [30, 40], [30, 40], True],   # refit again
        [2, [0, 30], [39, 40], [40, 50], [40, 50], False],  # no refit
    ]
    
    metrics = [add_y_train_argument(mean_absolute_error)]
    
    results = _calculate_metrics_backtesting_multiseries(
        series=data,
        predictions=predictions,
        folds=folds_mixed_refit,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=levels,
        add_aggregated_metric=False,
    )

    # The metric calculation should work regardless of refit pattern
    assert results.shape == (3, 2)
    assert list(results.columns) == ["levels", "mean_absolute_error"]
    assert list(results["levels"]) == ["item_1", "item_2", "item_3"]
    assert results["mean_absolute_error"].notna().all()


def test_calculate_metrics_backtesting_multiseries_with_window_size_variation():
    """
    Test that _calculate_metrics_backtesting_multiseries correctly handles 
    different window sizes (which affect y_train exclusion).
    """
    metrics = [add_y_train_argument(mean_absolute_scaled_error)]
    
    # Test with larger window size
    results_ws5 = _calculate_metrics_backtesting_multiseries(
        series=data,
        predictions=predictions,
        folds=folds,
        span_index=span_index,
        window_size=5,  # larger window
        metrics=metrics,
        levels=levels,
        add_aggregated_metric=False,
    )

    results_ws2 = _calculate_metrics_backtesting_multiseries(
        series=data,
        predictions=predictions,
        folds=folds,
        span_index=span_index,
        window_size=2,  # original window
        metrics=metrics,
        levels=levels,
        add_aggregated_metric=False,
    )

    # Metrics should differ because y_train differs (fewer observations with larger window)
    # but both should be valid numbers
    assert results_ws5.shape == results_ws2.shape
    assert results_ws5["mean_absolute_scaled_error"].notna().all()
    assert results_ws2["mean_absolute_scaled_error"].notna().all()
    # Different window sizes should produce different scaled error values
    assert not np.allclose(
        results_ws5["mean_absolute_scaled_error"].values,
        results_ws2["mean_absolute_scaled_error"].values
    )


def test_calculate_metrics_backtesting_multiseries_with_nans_in_series():
    """
    Test that _calculate_metrics_backtesting_multiseries correctly handles 
    series with NaN values.
    """
    # Create data with some NaN values
    data_with_nans = data.copy()
    data_with_nans.iloc[30:35, 0] = np.nan  # NaNs in item_1
    
    metrics = [add_y_train_argument(mean_absolute_error)]
    
    # This should work without raising errors
    results = _calculate_metrics_backtesting_multiseries(
        series=data_with_nans,
        predictions=predictions,
        folds=folds,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=levels,
        add_aggregated_metric=True,
    )

    # Should still produce valid output structure
    assert results.shape[0] == 6  # 3 levels + 3 aggregations
    assert results.shape[1] == 2  # levels + 1 metric


def test_calculate_metrics_backtesting_multiseries_preserves_level_order():
    """
    Test that _calculate_metrics_backtesting_multiseries preserves the order
    of levels as provided in the input.
    """
    # Reverse the order of levels
    reversed_levels = ["item_3", "item_2", "item_1"]
    
    metrics = [add_y_train_argument(mean_absolute_error)]
    
    results = _calculate_metrics_backtesting_multiseries(
        series=data,
        predictions=predictions,
        folds=folds,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=reversed_levels,
        add_aggregated_metric=False,
    )

    # Order should match input
    assert list(results["levels"]) == reversed_levels


# Tests for DataFrame without MultiIndex (stack optimization)
# ==============================================================================
def test_calculate_metrics_backtesting_multiseries_dataframe_no_multiindex_vs_dict():
    """
    Test that results are identical whether series is provided as a DataFrame
    without MultiIndex or as a dictionary. This validates the stack() 
    optimization produces the same results as pd.concat() for dicts.
    """
    # DataFrame without MultiIndex
    series_df = data.copy()
    assert not isinstance(series_df.index, pd.MultiIndex)
    
    # Convert to dict
    series_dict = {col: data[col] for col in data.columns}
    
    metrics = [add_y_train_argument(mean_absolute_error), add_y_train_argument(mean_absolute_scaled_error)]
    
    results_df = _calculate_metrics_backtesting_multiseries(
        series=series_df,
        predictions=predictions,
        folds=folds,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=levels,
        add_aggregated_metric=True,
    )
    
    results_dict = _calculate_metrics_backtesting_multiseries(
        series=series_dict,
        predictions=predictions,
        folds=folds,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=levels,
        add_aggregated_metric=True,
    )

    pd.testing.assert_frame_equal(results_df, results_dict)


def test_calculate_metrics_backtesting_multiseries_dataframe_no_multiindex_two_levels():
    """
    Test with DataFrame without MultiIndex containing only 2 levels.
    """
    # Select only 2 columns
    data_2_levels = data[["item_1", "item_2"]].copy()
    levels_2 = ["item_1", "item_2"]
    
    # Filter predictions for these levels
    predictions_2_levels = predictions.query("level in ['item_1', 'item_2']").copy()
    
    metrics = [add_y_train_argument(mean_absolute_error)]
    
    results = _calculate_metrics_backtesting_multiseries(
        series=data_2_levels,
        predictions=predictions_2_levels,
        folds=folds,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=levels_2,
        add_aggregated_metric=True,
    )

    # Should have 2 levels + 3 aggregations = 5 rows
    assert results.shape[0] == 5
    assert list(results["levels"]) == ["item_1", "item_2", "average", "weighted_average", "pooling"]
    
    # Verify individual metrics match expected values
    assert np.isclose(results[results["levels"] == "item_1"]["mean_absolute_error"].values[0], 1.477567, rtol=1e-5)
    assert np.isclose(results[results["levels"] == "item_2"]["mean_absolute_error"].values[0], 3.480129, rtol=1e-5)


def test_calculate_metrics_backtesting_multiseries_dataframe_no_multiindex_column_order():
    """
    Test that the stack() optimization preserves correct level-value mapping
    regardless of column order in the DataFrame.
    """
    # Reorder columns
    data_reordered = data[["item_3", "item_1", "item_2"]].copy()
    
    metrics = [add_y_train_argument(mean_absolute_error)]
    
    results = _calculate_metrics_backtesting_multiseries(
        series=data_reordered,
        predictions=predictions,
        folds=folds,
        span_index=span_index,
        window_size=window_size,
        metrics=metrics,
        levels=levels,  # original order
        add_aggregated_metric=False,
    )

    # Metrics should be the same regardless of column order in source DataFrame
    expected = pd.DataFrame(
        data={
            "levels": ["item_1", "item_2", "item_3"],
            "mean_absolute_error": [1.477567, 3.480129, 2.942386],
        }
    )

    pd.testing.assert_frame_equal(results, expected)


def test_calculate_metrics_backtesting_multiseries_dataframe_no_multiindex_many_levels():
    """
    Test DataFrame without MultiIndex with many levels (5+).
    """
    # Create data with 5 levels
    np.random.seed(123)
    n_rows = 50
    data_many = pd.DataFrame(
        {f"level_{i}": np.random.randn(n_rows).cumsum() + 20 for i in range(1, 6)},
        index=pd.date_range(start="2012-01-01", periods=n_rows)
    )
    
    assert not isinstance(data_many.index, pd.MultiIndex)
    
    # Create predictions for all levels
    predictions_many = pd.DataFrame(
        {f"level_{i}": np.random.randn(25).cumsum() + 22 for i in range(1, 6)},
        index=pd.date_range(start="2012-01-26", periods=25)
    )
    predictions_many = (
        predictions_many.melt(var_name="level", value_name="pred", ignore_index=False)
        .reset_index()
        .sort_values(by=["index", "level"])
        .set_index("index")
        .rename_axis('idx', axis=0)
        .set_index('level', append=True)
    )
    
    levels_many = [f"level_{i}" for i in range(1, 6)]
    
    metrics = [add_y_train_argument(mean_absolute_error)]
    
    results = _calculate_metrics_backtesting_multiseries(
        series=data_many,
        predictions=predictions_many,
        folds=folds,
        span_index=pd.date_range(start="2012-01-01", periods=n_rows),
        window_size=window_size,
        metrics=metrics,
        levels=levels_many,
        add_aggregated_metric=True,
    )

    # Should have 5 levels + 3 aggregations = 8 rows
    assert results.shape[0] == 8
    assert results["mean_absolute_error"].notna().all()
    assert list(results["levels"][:5]) == levels_many
    assert list(results["levels"][5:]) == ["average", "weighted_average", "pooling"]