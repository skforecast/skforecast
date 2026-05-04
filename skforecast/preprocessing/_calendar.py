################################################################################
#                               experimental                                   #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.validation import check_is_fitted


#TODO: add argument keep_original_columns=TRUE to indicate that new features are added to the already existent
def create_datetime_features(
    X: pd.Series | pd.DataFrame,
    features: list[str] | None = None,
    features_to_encode: list[str] | None = None,
    encoding: str = "cyclical",
    max_values: dict[str, int] | None = None,
    spline_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Extract datetime features from the DateTime index of a pandas DataFrame or Series.

    Parameters
    ----------
    X : pandas Series, pandas DataFrame
        Input DataFrame or Series with a datetime index.
    features : list, default None
        List of calendar features (strings) to extract from the index. When `None`,
        the following features are extracted: 'year', 'month', 'week', 'day_of_week',
        'day_of_month', 'day_of_year', 'weekend', 'hour', 'minute', 'second',
        'quarter'.
    features_to_encode : list, default None
        List of calendar features (strings) to encode. When `None`, all extracted
        features are encoded. If a feature is not in `features`, a ValueError is raised.
    encoding : str, default 'cyclical'
        Encoding method for the extracted features. Options are None, 'cyclical',
        'onehot' or 'spline'.
    max_values : dict, default None
        Dictionary of maximum values for the cyclical and spline encoding of calendar
        features. When `None`, the following values are used: {'month': 12, 'week': 52,
        'day_of_week': 7, 'day_of_month': 31, 'day_of_year': 365, 'hour': 24,
        'minute': 60, 'second': 60, 'quarter': 4}. Features not present in
        `max_values` (e.g. 'year', 'weekend') are left as raw integers when using
        spline encoding.
    spline_kwargs : dict, default None
        Additional keyword arguments for the spline encoding. Only used when
        `encoding='spline'`. When `None`, defaults to `{'degree': 3, 'include_bias':
        True}`; `n_knots` defaults to `max_values[feature] + 1` per feature, which
        produces one spline column per distinct period value (analogous to a smooth
        one-hot encoding). Knots are placed uniformly between the known minimum and
        maximum value of each feature (e.g. 1-12 for month, 0-23 for hour), making
        the encoding stateless and consistent between training and prediction. Accepted
        keys: `n_knots` (int), `degree` (int), and `include_bias` (bool).

    Returns
    -------
    X_new : pandas DataFrame
        DataFrame with the extracted (and optionally encoded) datetime features.
    
    """

    if not isinstance(X, (pd.DataFrame, pd.Series)):
        raise TypeError("Input `X` must be a pandas Series or DataFrame")
    if not isinstance(X.index, pd.DatetimeIndex):
        raise TypeError("Input `X` must have a pandas DatetimeIndex")
    if encoding not in ["cyclical", "onehot", "spline", None]:
        raise ValueError("Encoding must be one of 'cyclical', 'onehot', 'spline' or None")

    default_features = [
        "year",
        "month",
        "week",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "weekend",
        "hour",
        "minute",
        "second",
        "quarter",
    ]
    if features is None:
        features = default_features

    default_max_values = {
        "month": 12,
        "week": 52,
        "day_of_week": 7,
        "day_of_month": 31,
        "day_of_year": 365,
        "hour": 24,
        "minute": 60,
        "second": 60,
        "quarter": 4,
    }
    if max_values is None:
        max_values = default_max_values

    X_new = pd.DataFrame(index=X.index)

    datetime_attrs = {
        "year": "year",
        "month": "month",
        "week": lambda idx: idx.isocalendar().week.astype(int),
        "day_of_week": "dayofweek",
        "day_of_year": "dayofyear",
        "day_of_month": "day",
        "weekend": lambda idx: (idx.weekday >= 5).astype(int),
        "hour": "hour",
        "minute": "minute",
        "second": "second",
        "quarter": "quarter",
    }

    not_supported_features = set(features) - set(datetime_attrs.keys())
    if not_supported_features:
        raise ValueError(
            f"Features {not_supported_features} are not supported. "
            f"Supported features are {list(datetime_attrs.keys())}."
        )

    for feature in features:
        attr = datetime_attrs[feature]
        X_new[feature] = (
            attr(X.index) if callable(attr) else getattr(X.index, attr).astype(int)
        )

    if features_to_encode is not None:
        not_supported_features_to_encode = set(features_to_encode) - set(features)
        if not_supported_features_to_encode:
            raise ValueError(
                f"Features {not_supported_features_to_encode} are not present in `features`."
            )
    else:
        features_to_encode = features

    if encoding == "cyclical":
        cols_to_drop = []
        for feature, max_val in max_values.items():
            if feature in X_new.columns and feature in features_to_encode:
                X_new[f"{feature}_sin"] = np.sin(2 * np.pi * X_new[feature] / max_val)
                X_new[f"{feature}_cos"] = np.cos(2 * np.pi * X_new[feature] / max_val)
                cols_to_drop.append(feature)
        X_new = X_new.drop(columns=cols_to_drop)
    elif encoding == "onehot":
        X_new = pd.get_dummies(
            X_new, columns=features_to_encode, drop_first=False, sparse=False, dtype=int
        )
    elif encoding == "spline":
        resolved_spline_kwargs = {"degree": 3, "include_bias": True}
        if spline_kwargs is not None:
            resolved_spline_kwargs.update(spline_kwargs)
        degree = resolved_spline_kwargs["degree"]
        include_bias = resolved_spline_kwargs["include_bias"]
        n_knots_global = resolved_spline_kwargs.get("n_knots", None)
        default_min_values = {
            "month": 1,
            "week": 1,
            "day_of_week": 0,
            "day_of_month": 1,
            "day_of_year": 1,
            "hour": 0,
            "minute": 0,
            "second": 0,
            "quarter": 1,
        }
        cols_to_drop = []
        spline_cols = {}
        for feature, max_val in max_values.items():
            if feature in X_new.columns and feature in features_to_encode:
                n_knots = n_knots_global if n_knots_global is not None else max_val + 1
                min_val = default_min_values.get(feature, 0)
                knots = np.linspace(min_val, max_val, n_knots).reshape(-1, 1)
                spt = SplineTransformer(
                    degree=degree,
                    knots=knots,
                    extrapolation="periodic",
                    include_bias=include_bias,
                )
                values = X_new[feature].to_numpy().reshape(-1, 1)
                spline_out = spt.fit_transform(values)
                col_names = spt.get_feature_names_out([feature])
                for col_name, col_values in zip(col_names, spline_out.T):
                    spline_cols[col_name] = col_values
                cols_to_drop.append(feature)
        X_new = X_new.drop(columns=cols_to_drop)
        X_new = pd.concat(
            [X_new, pd.DataFrame(spline_cols, index=X_new.index)], axis=1
        )

    return X_new


#TODO: add argument keep_original_columns=TRUE to indicate that new features are added to the already existent
class DateTimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for extracting datetime features from the DateTime index of a
    pandas DataFrame or Series. It can also apply encoding to the extracted features.

    Parameters
    ----------
    features : list, default None
        List of calendar features (strings) to extract from the index. When `None`,
        the following features are extracted: 'year', 'month', 'week', 'day_of_week',
        'day_of_month', 'day_of_year', 'weekend', 'hour', 'minute', 'second',
        'quarter'. Additional supported features that are not extracted by default
        can be passed explicitly.
    features_to_encode : list, default None
        List of calendar features (strings) to encode. When `None`, all extracted
        features are encoded. If a feature is not in `features`, a ValueError is raised.
    encoding : str, default 'cyclical'
        Encoding method for the extracted features. Options are None, 'cyclical',
        'onehot' or 'spline'.
    max_values : dict, default None
        Dictionary of maximum values for the cyclical and spline encoding of calendar
        features. When `None`, the following values are used: {'month': 12, 'week': 52,
        'day_of_week': 7, 'day_of_month': 31, 'day_of_year': 365, 'hour': 24,
        'minute': 60, 'second': 60, 'quarter': 4}.
    spline_kwargs : dict, default None
        Additional keyword arguments for the spline encoding. Only used when
        `encoding='spline'`. When `None`, defaults to `{'degree': 3, 'include_bias':
        True}`; `n_knots` defaults to `max_values[feature] + 1` per feature. Knots
        are placed uniformly over the known range of each feature (e.g. 1-12 for
        month, 0-23 for hour), ensuring consistent encoding across training and
        prediction. Accepted keys: `n_knots` (int), `degree` (int), and
        `include_bias` (bool).
    
    Attributes
    ----------
    features : list, None
        List of calendar features to extract from the index. `None` means the
        default features are used.
    features_to_encode : list, None
        List of calendar features to encode. `None` means all extracted features are encoded.
    encoding : str
        Encoding method for the extracted features.
    max_values : dict, None
        Dictionary of maximum values for the cyclical and spline encoding of calendar
        features. `None` means the default values are used.
    spline_kwargs : dict, None
        Keyword arguments for the spline encoding. `None` means the default values
        are used (`degree=3`, `include_bias=True`, `n_knots=max_val+1` per feature).
    feature_names_out_ : list
        Names of the output features. Set after calling `transform`.
    
    """

    def __init__(
        self,
        features: list[str] | None = None,
        features_to_encode: list[str] | None = None,
        encoding: str = "cyclical",
        max_values: dict[str, int] | None = None,
        spline_kwargs: dict | None = None,
    ) -> None:

        if encoding not in ["cyclical", "onehot", "spline", None]:
            raise ValueError("Encoding must be one of 'cyclical', 'onehot', 'spline' or None")

        self.features = features
        self.features_to_encode = features_to_encode
        self.encoding = encoding
        self.max_values = max_values
        self.spline_kwargs = spline_kwargs

    def fit(self, X, y=None):
        """
        A no-op method to satisfy the scikit-learn API.
        """
        return self

    def transform(
        self,
        X: pd.Series | pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create datetime features from the DateTime index of a pandas DataFrame or Series.

        Parameters
        ----------
        X : pandas Series, pandas DataFrame
            Input DataFrame or Series with a datetime index.
        
        Returns
        -------
        X_new : pandas DataFrame
            DataFrame with the extracted (and optionally encoded) datetime features.

        """

        X_new = create_datetime_features(
                    X                  = X,
                    encoding           = self.encoding,
                    features           = self.features,
                    features_to_encode = self.features_to_encode,
                    max_values         = self.max_values,
                    spline_kwargs      = self.spline_kwargs,
                )
        self.feature_names_out_ = list(X_new.columns)

        return X_new

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None
    ) -> list[str]:
        """
        Get the names of the output features.

        Parameters
        ----------
        input_features : list, default None
            Ignored. Present for API compatibility with sklearn.

        Returns
        -------
        feature_names_out_ : list
            Names of the output features.

        """
        check_is_fitted(self, "feature_names_out_")

        return self.feature_names_out_

# TODO: if date_column is None use the index
def calculate_distance_from_holiday(
    df: pd.DataFrame, 
    holiday_column: str = 'is_holiday',
    date_column: str = 'date', 
    fill_na: int | float = 0.
) -> pd.DataFrame:  # pragma: no cover
    """
    Calculate the number of days to the next holiday and the number of days since 
    the last holiday.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the holiday data.
    holiday_column : str, default 'is_holiday'
        The name of the column indicating holidays (True/False), by default 'is_holiday'.
    date_column : str, default 'date'
        The name of the column containing the dates, by default 'date'.
    fill_na : int, float, default 0.
        Value to fill for NaN values in the output columns, by default 0.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for days to the next holiday ('days_to_holiday') 
        and days since the last holiday ('days_since_holiday').
    
    Notes
    -----
    The function assumes that the input `df` contains a boolean column indicating holidays
    and a date column. It calculates the number of days to the next holiday and the number of
    days since the last holiday for each date in the date column.

    """

    df = df.reset_index(drop=True)
    df[date_column] = pd.to_datetime(df[date_column])
    
    dates = df[date_column].to_numpy()
    holiday_dates = df.loc[df[holiday_column], date_column].to_numpy()
    holiday_dates_sorted = np.sort(holiday_dates)

    # For next holiday (right side)
    next_idx = np.searchsorted(holiday_dates_sorted, dates, side='left')
    has_next = next_idx < len(holiday_dates_sorted)
    days_to_holiday = np.full(len(dates), np.nan)
    days_to_holiday[has_next] = (
        holiday_dates_sorted[next_idx[has_next]] - dates[has_next]
    ).astype('timedelta64[D]').astype(int)

    # For previous holiday (left side)
    prev_idx = np.searchsorted(holiday_dates_sorted, dates, side='right') - 1
    has_prev = prev_idx >= 0
    days_since_holiday = np.full(len(dates), np.nan)
    days_since_holiday[has_prev] = (
        dates[has_prev] - holiday_dates_sorted[prev_idx[has_prev]]
    ).astype('timedelta64[D]').astype(int)

    df["days_to_holiday"] = pd.Series(days_to_holiday, dtype="Int64").fillna(fill_na)
    df["days_since_holiday"] = pd.Series(days_since_holiday, dtype="Int64").fillna(fill_na)
    
    return df
