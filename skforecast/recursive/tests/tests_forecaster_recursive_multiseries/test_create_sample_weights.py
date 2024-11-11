# Unit test create_sample_weights ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from ....recursive import ForecasterRecursiveMultiSeries


def custom_weights(index):  # pragma: no cover
    """
    Return 0 if index is between '2022-01-08' and '2022-01-10', 1 otherwise.
    """
    weights = np.where((index >= "2022-01-08") & (index <= "2022-01-10"), 0, 1)

    return weights


def custom_weights_2(index):  # pragma: no cover
    """
    Return 2 if index is between '2022-01-11' and '2022-01-13', 3 otherwise.
    """
    weights = np.where((index >= "2022-01-11") & (index <= "2022-01-13"), 2, 3)

    return weights


def custom_weights_nan(index):  # pragma: no cover
    """
    Return np.nan if index is between '2022-01-08' and '2022-01-10', 1 otherwise.
    """
    weights = np.where((index >= "2022-01-08") & (index <= "2022-01-10"), np.nan, 1)

    return weights


def custom_weights_negative(index):  # pragma: no cover
    """
    Return -1 if index is between '2022-01-08' and '2022-01-10', 1 otherwise.
    """
    weights = np.where((index >= "2022-01-08") & (index <= "2022-01-10"), -1, 1)

    return weights


series = pd.DataFrame(
    data=np.array(
        [
            [0.12362923, 0.51328688],
            [0.65138268, 0.11599708],
            [0.58142898, 0.72350895],
            [0.72969992, 0.10305721],
            [0.97790567, 0.20581485],
            [0.56924731, 0.41262027],
            [0.85369084, 0.82107767],
            [0.75425194, 0.0107816],
            [0.08167939, 0.94951918],
            [0.00249297, 0.55583355],
        ]
    ),
    columns=["series_1", "series_2"],
    index=pd.DatetimeIndex(
        [
            "2022-01-04",
            "2022-01-05",
            "2022-01-06",
            "2022-01-07",
            "2022-01-08",
            "2022-01-09",
            "2022-01-10",
            "2022-01-11",
            "2022-01-12",
            "2022-01-13",
        ],
        dtype="datetime64[ns]",
        freq="D",
    ),
)

X_train_onehot = pd.DataFrame(
    data=np.array(
        [
            [0.58142898, 0.65138268, 0.12362923, 1.0, 0.0],
            [0.72969992, 0.58142898, 0.65138268, 1.0, 0.0],
            [0.97790567, 0.72969992, 0.58142898, 1.0, 0.0],
            [0.56924731, 0.97790567, 0.72969992, 1.0, 0.0],
            [0.85369084, 0.56924731, 0.97790567, 1.0, 0.0],
            [0.75425194, 0.85369084, 0.56924731, 1.0, 0.0],
            [0.08167939, 0.75425194, 0.85369084, 1.0, 0.0],
            [0.72350895, 0.11599708, 0.51328688, 0.0, 1.0],
            [0.10305721, 0.72350895, 0.11599708, 0.0, 1.0],
            [0.20581485, 0.10305721, 0.72350895, 0.0, 1.0],
            [0.41262027, 0.20581485, 0.10305721, 0.0, 1.0],
            [0.82107767, 0.41262027, 0.20581485, 0.0, 1.0],
            [0.0107816, 0.82107767, 0.41262027, 0.0, 1.0],
            [0.94951918, 0.0107816, 0.82107767, 0.0, 1.0],
        ]
    ),
    columns=["lag_1", "lag_2", "lag_3", "series_1", "series_2"],
    index=pd.DatetimeIndex(
        [
            "2022-01-07",
            "2022-01-08",
            "2022-01-09",
            "2022-01-10",
            "2022-01-11",
            "2022-01-12",
            "2022-01-13",
            "2022-01-07",
            "2022-01-08",
            "2022-01-09",
            "2022-01-10",
            "2022-01-11",
            "2022-01-12",
            "2022-01-13",
        ],
        dtype="datetime64[ns]",
        freq=None,
    ),
)

X_train_ordinal = pd.DataFrame(
    data=np.array(
        [
            [0.58142898, 0.65138268, 0.12362923, 0.0],
            [0.72969992, 0.58142898, 0.65138268, 0.0],
            [0.97790567, 0.72969992, 0.58142898, 0.0],
            [0.56924731, 0.97790567, 0.72969992, 0.0],
            [0.85369084, 0.56924731, 0.97790567, 0.0],
            [0.75425194, 0.85369084, 0.56924731, 0.0],
            [0.08167939, 0.75425194, 0.85369084, 0.0],
            [0.72350895, 0.11599708, 0.51328688, 1.0],
            [0.10305721, 0.72350895, 0.11599708, 1.0],
            [0.20581485, 0.10305721, 0.72350895, 1.0],
            [0.41262027, 0.20581485, 0.10305721, 1.0],
            [0.82107767, 0.41262027, 0.20581485, 1.0],
            [0.0107816, 0.82107767, 0.41262027, 1.0],
            [0.94951918, 0.0107816, 0.82107767, 1.0],
        ]
    ),
    columns=["lag_1", "lag_2", "lag_3", "_level_skforecast"],
    index=pd.DatetimeIndex(
        [
            "2022-01-07",
            "2022-01-08",
            "2022-01-09",
            "2022-01-10",
            "2022-01-11",
            "2022-01-12",
            "2022-01-13",
            "2022-01-07",
            "2022-01-08",
            "2022-01-09",
            "2022-01-10",
            "2022-01-11",
            "2022-01-12",
            "2022-01-13",
        ],
        dtype="datetime64[ns]",
        freq=None,
    ),
)

X_train_ordinal_category = X_train_ordinal.copy()
X_train_ordinal_category["_level_skforecast"] = X_train_ordinal_category[
    "_level_skforecast"
].astype("category")

X_train_onehot_diferent_length = pd.DataFrame(
    data=np.array(
        [
            [0.58142898, 0.65138268, 0.12362923, 1.0, 0.0],
            [0.72969992, 0.58142898, 0.65138268, 1.0, 0.0],
            [0.97790567, 0.72969992, 0.58142898, 1.0, 0.0],
            [0.56924731, 0.97790567, 0.72969992, 1.0, 0.0],
            [0.85369084, 0.56924731, 0.97790567, 1.0, 0.0],
            [0.75425194, 0.85369084, 0.56924731, 1.0, 0.0],
            [0.08167939, 0.75425194, 0.85369084, 1.0, 0.0],
            [0.41262027, 0.20581485, 0.10305721, 0.0, 1.0],
            [0.82107767, 0.41262027, 0.20581485, 0.0, 1.0],
            [0.0107816, 0.82107767, 0.41262027, 0.0, 1.0],
            [0.94951918, 0.0107816, 0.82107767, 0.0, 1.0],
        ]
    ),
    columns=["lag_1", "lag_2", "lag_3", "series_1", "series_2"],
    index=pd.DatetimeIndex(
        [
            "2022-01-07",
            "2022-01-08",
            "2022-01-09",
            "2022-01-10",
            "2022-01-11",
            "2022-01-12",
            "2022-01-13",
            "2022-01-10",
            "2022-01-11",
            "2022-01-12",
            "2022-01-13",
        ],
        dtype="datetime64[ns]",
        freq=None,
    ),
)

X_train_ordinal_diferent_length = pd.DataFrame(
    data=np.array(
        [
            [0.58142898, 0.65138268, 0.12362923, 0.0],
            [0.72969992, 0.58142898, 0.65138268, 0.0],
            [0.97790567, 0.72969992, 0.58142898, 0.0],
            [0.56924731, 0.97790567, 0.72969992, 0.0],
            [0.85369084, 0.56924731, 0.97790567, 0.0],
            [0.75425194, 0.85369084, 0.56924731, 0.0],
            [0.08167939, 0.75425194, 0.85369084, 0.0],
            [0.41262027, 0.20581485, 0.10305721, 1.0],
            [0.82107767, 0.41262027, 0.20581485, 1.0],
            [0.0107816, 0.82107767, 0.41262027, 1.0],
            [0.94951918, 0.0107816, 0.82107767, 1.0],
        ]
    ),
    columns=["lag_1", "lag_2", "lag_3", "_level_skforecast"],
    index=pd.DatetimeIndex(
        [
            "2022-01-07",
            "2022-01-08",
            "2022-01-09",
            "2022-01-10",
            "2022-01-11",
            "2022-01-12",
            "2022-01-13",
            "2022-01-10",
            "2022-01-11",
            "2022-01-12",
            "2022-01-13",
        ],
        dtype="datetime64[ns]",
        freq=None,
    ),
)

X_train_ordinal_category_diferent_length = X_train_ordinal_diferent_length.copy()
X_train_ordinal_category_diferent_length["_level_skforecast"] = (
    X_train_ordinal_category_diferent_length["_level_skforecast"].astype("category")
)


@pytest.mark.parametrize(
    "encoding, X_train",
    [
        ("ordinal", X_train_ordinal),
        ("ordinal_category", X_train_ordinal_category),
        ("onehot", X_train_onehot),
        (None, X_train_ordinal)
    ],
    ids=lambda dt: f"encoding, X_train: {dt}",
)
def test_create_sample_weights_output_using_series_weights(encoding, X_train):
    """
    Test `sample_weights` creation using `series_weights`.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=LinearRegression(),
        lags=3,
        encoding=encoding,
        series_weights={"series_1": 1.0, "series_2": 2.0},
    )
    forecaster.encoding_mapping_ = {"series_1": 0, "series_2": 1}

    results = forecaster.create_sample_weights(
        series_names_in_=["series_1", "series_2"], X_train=X_train
    )

    expected = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    )

    assert np.array_equal(results, expected)


@pytest.mark.parametrize(
    "encoding, X_train",
    [
        ("ordinal", X_train_ordinal),
        ("ordinal_category", X_train_ordinal_category),
        ("onehot", X_train_onehot),
        (None, X_train_ordinal)
    ],
    ids=lambda dt: f"encoding, X_train: {dt}",
)
def test_create_sample_weights_output_using_weight_func(encoding, X_train):
    """
    Test `sample_weights` creation using `weight_func`.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=LinearRegression(),
        lags=3,
        encoding=encoding,
        weight_func=custom_weights,
    )
    forecaster.encoding_mapping_ = {"series_1": 0, "series_2": 1}
    results = forecaster.create_sample_weights(
        series_names_in_=["series_1", "series_2"],
        X_train=X_train,
    )
    expected = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1])

    assert np.array_equal(results, expected)


@pytest.mark.parametrize(
    "weight_func, expected",
    [
        (
            {"series_1": custom_weights},
            np.array(
                [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ),
        ),
        (
            {"series_2": custom_weights_2},
            np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0]
            ),
        ),
        (
            {"series_1": custom_weights, "series_2": custom_weights_2},
            np.array([1, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2]),
        ),
    ],
    ids=lambda values: f"levels: {values}",
)
def test_create_sample_weights_output_using_weight_func_dict(weight_func, expected):
    """
    Test `sample_weights` creation using `weight_func`.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     encoding           = "ordinal",
                     transformer_series = StandardScaler(),
                     weight_func        = weight_func
                 )
    forecaster.encoding_mapping_ = {"series_1": 0, "series_2": 1}
    results = forecaster.create_sample_weights(
        series_names_in_=["series_1", "series_2"], X_train=X_train_ordinal
    )

    assert np.array_equal(results, expected)


@pytest.mark.parametrize(
    "weight_func, expected",
    [
        (
            {"series_1": custom_weights},
            np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
        ),
        (
            {"series_2": custom_weights_2},
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 2.0, 2.0, 2.0]),
        ),
        (
            {"series_1": custom_weights, "series_2": custom_weights_2},
            np.array([1, 0, 0, 0, 1, 1, 1, 3, 2, 2, 2]),
        ),
    ],
    ids=lambda values: f"levels: {values}",
)
def test_create_sample_weights_output_using_weight_func_dict_different_series_lengths(
    weight_func, expected
):
    """
    Test `sample_weights` creation using `weight_func` with series of different lengths.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     encoding           = "ordinal",
                     transformer_series = StandardScaler(),
                     weight_func        = weight_func
                 )
    forecaster.encoding_mapping_ = {"series_1": 0, "series_2": 1}
    results = forecaster.create_sample_weights(
        series_names_in_=["series_1", "series_2"],
        X_train=X_train_ordinal_diferent_length,
    )

    assert np.array_equal(results, expected)


@pytest.mark.parametrize(
    "encoding, X_train",
    [
        ("ordinal", X_train_ordinal),
        ("ordinal_category", X_train_ordinal_category),
        ("onehot", X_train_onehot),
        (None, X_train_ordinal)
    ],
    ids=lambda dt: f"encoding, X_train: {dt}",
)
def test_create_sample_weights_output_using_series_weights_and_weight_func(
    encoding, X_train
):
    """
    Test `sample_weights` creation using `series_weights` and `weight_func`.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     encoding           = encoding,
                     transformer_series = StandardScaler(),
                     series_weights     = {"series_1": 1.0, "series_2": 2.0},
                     weight_func        = custom_weights
                 )
    forecaster.encoding_mapping_ = {"series_1": 0, "series_2": 1}
    results = forecaster.create_sample_weights(
        series_names_in_=["series_1", "series_2"], X_train=X_train
    )

    expected = np.array([1, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 2, 2, 2], dtype=float)

    assert np.array_equal(results, expected)


@pytest.mark.parametrize(
    "encoding, X_train",
    [
        ("ordinal", X_train_ordinal_diferent_length),
        ("ordinal_category", X_train_ordinal_category_diferent_length),
        ("onehot", X_train_onehot_diferent_length),
        (None, X_train_ordinal_diferent_length)
    ],
    ids=lambda dt: f"encoding, X_train: {dt}",
)
def test_create_sample_weights_output_using_series_weights_and_weight_func_different_series_lengths(
    encoding, X_train
):
    """
    Test `sample_weights` creation using `series_weights` and `weight_func`
    with series of different lengths.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     encoding           = encoding,
                     transformer_series = StandardScaler(),
                     series_weights     = {"series_1": 1.0, "series_2": 2.0},
                     weight_func        = custom_weights
                 )
    forecaster.encoding_mapping_ = {"series_1": 0, "series_2": 1}
    results = forecaster.create_sample_weights(
        series_names_in_=["series_1", "series_2"], X_train=X_train
    )
    
    expected = np.array([1, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2], dtype=float)

    assert np.array_equal(results, expected)


@pytest.mark.parametrize(
    "encoding, X_train",
    [
        ("ordinal", X_train_ordinal),
        ("ordinal_category", X_train_ordinal_category),
        ("onehot", X_train_onehot),
        (None, X_train_ordinal)
    ],
    ids=lambda dt: f"encoding, X_train: {dt}",
)
def test_create_sample_weights_ValueError_when_weights_has_nan(encoding, X_train):
    """
    Test sample_weights ValueError when sample_weight contains NaNs.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     encoding           = encoding,
                     transformer_series = StandardScaler(),
                     weight_func        = custom_weights_nan
                 )
    forecaster.encoding_mapping_ = {"series_1": 0, "series_2": 1}

    err_msg = re.escape("The resulting `weights` cannot have NaN values.")
    with pytest.raises(ValueError, match=err_msg):
        forecaster.create_sample_weights(
            series_names_in_=["series_1", "series_2"], X_train=X_train
        )


@pytest.mark.parametrize(
    "encoding, X_train",
    [
        ("ordinal", X_train_ordinal),
        ("ordinal_category", X_train_ordinal_category),
        ("onehot", X_train_onehot),
        (None, X_train_ordinal)
    ],
    ids=lambda dt: f"encoding, X_train: {dt}",
)
def test_create_sample_weights_ValueError_when_weights_has_negative_values(
    encoding, X_train
):
    """
    Test sample_weights ValueError when sample_weight contains negative values.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     encoding           = encoding,
                     transformer_series = StandardScaler(),
                     weight_func        = custom_weights_negative
                 )
    forecaster.encoding_mapping_ = {"series_1": 0, "series_2": 1}

    err_msg = re.escape("The resulting `weights` cannot have negative values.")
    with pytest.raises(ValueError, match=err_msg):
        forecaster.create_sample_weights(
            series_names_in_=["series_1", "series_2"], X_train=X_train
        )


@pytest.mark.parametrize(
    "encoding, X_train",
    [
        ("ordinal", X_train_ordinal),
        ("ordinal_category", X_train_ordinal_category),
        ("onehot", X_train_onehot),
        (None, X_train_ordinal)
    ],
    ids=lambda dt: f"encoding, X_train: {dt}",
)
def test_create_sample_weights_ValueError_when_weights_all_zeros(encoding, X_train):
    """
    Test sample_weights ValueError when the sum of the weights is zero.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     encoding           = encoding,
                     transformer_series = StandardScaler(),
                     series_weights     = {"series_1": 0, "series_2": 0},
                 )
    forecaster.encoding_mapping_ = {"series_1": 0, "series_2": 1}

    err_msg = re.escape(
        ("The resulting `weights` cannot be normalized because "
         "the sum of the weights is zero.")
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.create_sample_weights(
            series_names_in_=["series_1", "series_2"], X_train=X_train
        )
