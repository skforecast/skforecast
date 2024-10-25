# Unit test calculate_metrics_multiseries
# ==============================================================================
import pandas as pd
import numpy as np
import pytest as pytest
from skforecast.model_selection._utils import _calculate_metrics_multiseries
from skforecast.metrics import add_y_train_argument
from sklearn.metrics import mean_absolute_error
from skforecast.metrics import mean_absolute_scaled_error

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

predictions_missing_level = predictions.drop(columns="item_3").copy()

predictions_different_lenght = pd.DataFrame(
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

span_index = span_index = pd.date_range(start="2012-01-01", end="2012-02-19", freq="D")

folds = [
    [[0, 25], [24, 25], [25, 35], [25, 35], False],
    [[0, 25], [34, 35], [35, 45], [35, 45], False],
    [[0, 25], [44, 45], [45, 50], [45, 50], False],
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


def test_calculate_metrics_multiseries_input_types():
    """
    Check if function raises errors when input parameters have wrong types.
    """

    # Mock inputs
    series_df = pd.DataFrame(
        {"time": pd.date_range(start="2020-01-01", periods=3), "value": [1, 2, 3]}
    )
    predictions = pd.DataFrame(
        {
            "time": pd.date_range(start="2020-01-01", periods=3),
            "predicted": [1.5, 2.5, 3.5],
        }
    )
    folds = [{"train": (0, 1), "test": (2, 3)}]
    span_index = pd.date_range(start="2020-01-01", periods=3)
    window_size = 2
    metrics = ["mean_absolute_error"]
    levels = ["level1", "level2"]

    # Test invalid type for series
    msg = "`series` must be a pandas DataFrame or a dictionary of pandas DataFrames."
    with pytest.raises(TypeError, match=msg):
        _calculate_metrics_multiseries(
            "invalid_series_type", predictions, folds, span_index, window_size, metrics, levels
        )

    # Test invalid type for predictions
    msg = "`predictions` must be a pandas DataFrame."
    with pytest.raises(TypeError, match=msg):
        _calculate_metrics_multiseries(
            series_df, "invalid_predictions_type", folds, span_index, window_size, metrics, levels
        )

    # Test invalid type for folds
    msg = "`folds` must be a list."
    with pytest.raises(TypeError, match=msg):
        _calculate_metrics_multiseries(
            series_df, predictions, "invalid_folds_type", span_index, window_size, metrics, levels
        )

    # Test invalid type for span_index
    msg = "`span_index` must be a pandas DatetimeIndex or pandas RangeIndex."
    with pytest.raises(TypeError, match=msg):
        _calculate_metrics_multiseries(
            series_df, predictions, folds, "invalid_span_index_type", window_size, metrics, levels
        )

    # Test invalid type for window_size
    msg = "`window_size` must be an integer."
    with pytest.raises(TypeError, match=msg):
        _calculate_metrics_multiseries(
            series_df, predictions, folds, span_index, "invalid_window_size_type", metrics, levels
        )

    # Test invalid type for metrics
    msg = "`metrics` must be a list."
    with pytest.raises(TypeError, match=msg):
        _calculate_metrics_multiseries(
            series_df, predictions, folds, span_index, window_size, "invalid_metrics_type", levels
        )

    # Test invalid type for levels
    msg = "`levels` must be a list."
    with pytest.raises(TypeError, match=msg):
        _calculate_metrics_multiseries(
            series_df, predictions, folds, span_index, window_size, metrics, "invalid_levels_type"
        )

    # Test invalid type for add_aggregated_metric
    msg = "`add_aggregated_metric` must be a boolean."
    with pytest.raises(TypeError, match=msg):
        _calculate_metrics_multiseries(
            series_df,
            predictions,
            folds,
            span_index,
            window_size,
            metrics,
            levels,
            add_aggregated_metric="invalid_type",
        )


def test_calculate_metrics_multiseries_output_when_no_aggregated_metric(
    metrics=[mean_absolute_error, mean_absolute_scaled_error]
):
    """
    Test output of _calculate_metrics_multiseries when add_aggregated_metric=False
    """

    metrics = [add_y_train_argument(metric) for metric in metrics]
    results = _calculate_metrics_multiseries(
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


def test_calculate_metrics_multiseries_output_when_aggregated_metric(
    metrics=[mean_absolute_error, mean_absolute_scaled_error]
):
    """
    Test output of _calculate_metrics_multiseries when add_aggregated_metric=True
    """

    metrics = [add_y_train_argument(metric) for metric in metrics]
    results = _calculate_metrics_multiseries(
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


def test_calculate_metrics_multiseries_output_when_aggregated_metric_and_customer_metric(
    metrics=[custom_metric],
):
    """
    Test output of _calculate_metrics_multiseries when add_aggregated_metric=True
    """

    metrics = [add_y_train_argument(metric) for metric in metrics]
    results = _calculate_metrics_multiseries(
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


def test_calculate_metrics_multiseries_output_when_aggregated_metric_and_predictions_have_different_length(
    metrics=[mean_absolute_error, mean_absolute_scaled_error]
):
    """
    """
    metrics = [add_y_train_argument(metric) for metric in metrics]
    results = _calculate_metrics_multiseries(
        series=data,
        predictions=predictions_different_lenght,
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


def test_calculate_metrics_multiseries_output_when_aggregated_metric_and_one_level_is_not_predicted(
    metrics=[mean_absolute_error, mean_absolute_scaled_error]
):
    """
    
    """
    metrics = [add_y_train_argument(metric) for metric in metrics]
    results = _calculate_metrics_multiseries(
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
