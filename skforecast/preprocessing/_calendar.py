################################################################################
#                                Calendar Features                             #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import inspect
import re
import warnings
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.validation import check_is_fitted

from ..exceptions import IgnoredArgumentWarning


_FEATURE_KNOWN_CATEGORIES = {
    "month": list(range(1, 13)),
    "week": list(range(1, 54)),  # 1..53 to handle ISO week 53; see Notes in create_datetime_features
    "day_of_week": list(range(0, 7)),
    "day_of_month": list(range(1, 32)),
    "day_of_year": list(range(1, 367)),  # 1..366 to handle leap years; see Notes in create_datetime_features
    "hour": list(range(0, 24)),
    "minute": list(range(0, 60)),
    "second": list(range(0, 60)),
    "quarter": list(range(1, 5)),
}
_DEFAULT_MAX_VALUES = {k: len(v) for k, v in _FEATURE_KNOWN_CATEGORIES.items()}
_DEFAULT_MIN_VALUES = {k: v[0] for k, v in _FEATURE_KNOWN_CATEGORIES.items()}


def create_datetime_features(
    X: pd.Series | pd.DataFrame,
    features: list[str] | None = None,
    features_to_encode: list[str] | None = None,
    encoding: str = "cyclical",
    max_values: dict[str, int] | None = None,
    spline_kwargs: dict | None = None,
    keep_original_columns: bool = True,
) -> pd.DataFrame:
    """
    Extract datetime features from the DateTime index of a pandas DataFrame or Series.

    Parameters
    ----------
    X : pandas Series, pandas DataFrame
        Input DataFrame or Series with a datetime index.
    features : list, default None
        List of calendar features (strings) to extract from the index. When
        `None`, the following features are extracted: `'year'`, `'month'`,
        `'week'`, `'day_of_week'`, `'day_of_month'`, `'day_of_year'`,
        `'weekend'`, `'hour'`, `'minute'`, `'second'`, `'quarter'`.
    features_to_encode : list, default None
        List of calendar features (strings) to encode. When `None`, all
        extracted features are encoded. If a feature is not in `features`, a
        `ValueError` is raised. If the explicit list contains features that
        cannot be encoded with the chosen `encoding` (e.g. `'year'` or
        `'weekend'`, which are never encodable), an `IgnoredArgumentWarning`
        is issued and those features are kept as raw integers.
    encoding : str, default 'cyclical'
        Encoding method for the extracted features. Options are `None`,
        `'cyclical'`, `'onehot'` or `'spline'`. Features that cannot be
        encoded under the chosen mode are kept as raw integers. By default,
        `'year'` and `'weekend'` are never encoded — `'onehot'` excludes them
        via the known-category set, while `'cyclical'` and `'spline'` exclude
        them via `max_values`.
    max_values : dict, default None
        Dictionary of maximum values for the cyclical and spline encoding.
        User-provided values are **merged** with the defaults: keys passed by
        the user override the corresponding default, and missing keys fall
        back to the defaults `{'month': 12, 'week': 53, 'day_of_week': 7,
        'day_of_month': 31, 'day_of_year': 366, 'hour': 24, 'minute': 60,
        'second': 60, 'quarter': 4}`. For example, passing
        `max_values={'month': 6}` overrides only `month`; the other features
        keep their defaults. Features that are not in the defaults (e.g.
        `'year'`, `'weekend'`) are left as raw integers.
    spline_kwargs : dict, default None
        Additional keyword arguments for the spline encoding. Only used when
        `encoding='spline'`. When `None`, defaults to `{'degree': 3,
        'include_bias': True, 'extrapolation': 'periodic'}`; `n_knots`
        defaults to `max_values[feature] + 1` per feature, which produces one
        spline column per distinct period value (analogous to a smooth
        one-hot encoding). Knots are placed uniformly between the known
        minimum and maximum value of each feature (e.g. 1-12 for month, 0-23
        for hour), making the encoding stateless and consistent between
        training and prediction. Any keyword argument accepted by
        `sklearn.preprocessing.SplineTransformer` is allowed (e.g. `n_knots`,
        `degree`, `include_bias`, `extrapolation`, `order`) **except**
        `knots` (computed internally from `max_values`) and `sparse_output`
        (incompatible with the DataFrame output). Passing either of these or
        an unknown key raises `ValueError`.
    keep_original_columns : bool, default True
        If True, the original columns of `X` are kept in the output
        DataFrame. If False, only the extracted datetime features are
        returned. When `True` and `X` is an unnamed pandas Series
        (`X.name is None`), a `ValueError` is raised; either set `X.name` to
        a string, or pass `keep_original_columns=False`.

    Returns
    -------
    X_new : pandas DataFrame
        DataFrame with the extracted (and optionally encoded) datetime features.

    Raises
    ------
    TypeError
        If `X` is not a pandas Series or DataFrame, or if its index is not a
        pandas DatetimeIndex.
    ValueError
        If `encoding` is not one of `'cyclical'`, `'onehot'`, `'spline'` or
        `None`; if `X` is an unnamed Series and `keep_original_columns=True`;
        if a feature in `features` is not supported; if a feature in
        `features_to_encode` is not present in `features`; if `spline_kwargs`
        contains a blocked key (`'knots'`, `'sparse_output'`) or an unknown
        key; or if extracted feature names overlap with existing columns in
        `X` when `keep_original_columns=True`.

    Warns
    -----
    IgnoredArgumentWarning
        When `features_to_encode` is explicitly passed and contains features
        that cannot be encoded with the chosen `encoding`. Those features
        are kept as raw integers.

    Notes
    -----
    The default `max_values` use 53 for `'week'` and 366 for `'day_of_year'`
    to accommodate the maximum possible values across all calendar years:
    ISO week 53 occurs in some years (e.g. 2015, 2020, 2026) and
    day-of-year 366 occurs in leap years. Because the encoding must be
    stateless (the same for any year, without prior knowledge of whether it
    is a leap year or contains ISO week 53), the period is fixed at the
    maximum-possible value. This implies:

    - **Onehot:** the `week_53` and `day_of_year_366` columns are always
      present in the output and equal 0 for rows whose year never reaches
      those values. This guarantees a consistent column schema across
      training and prediction.
    - **Cyclical / spline:** in years where the maximum value is reached,
      the cyclical wrap-around is exact (e.g. `sin(2π·366/366) = 0` matches
      `sin(2π·0/366) = 0`). In years where it is not, there is a one-step
      "phantom gap" between the highest observed value and 1 — the
      cyclical distance is two steps instead of one. This residual
      asymmetry is numerically small (≈ 1.7% for `day_of_year`, ≈ 12% for
      `week`) and is strictly preferable to the alternative (period
      52 / 365), which would silently collapse week 53 onto week 1 and day
      366 onto day 1 in years where those values occur.

    """

    if not isinstance(X, (pd.DataFrame, pd.Series)):
        raise TypeError("Input `X` must be a pandas Series or DataFrame")
    if not isinstance(X.index, pd.DatetimeIndex):
        raise TypeError("Input `X` must have a pandas DatetimeIndex")
    if encoding not in ["cyclical", "onehot", "spline", None]:
        raise ValueError("Encoding must be one of 'cyclical', 'onehot', 'spline' or None")
    if isinstance(X, pd.Series) and X.name is None and keep_original_columns:
        raise ValueError(
            "When `keep_original_columns=True`, the input Series must have a "
            "name (`X.name`). Either set `X.name` to a string, or pass "
            "`keep_original_columns=False`."
        )

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

    resolved_max_values = _DEFAULT_MAX_VALUES.copy()
    if max_values is not None:
        resolved_max_values.update(max_values)
    max_values = resolved_max_values

    X_new = pd.DataFrame(index=X.index)

    datetime_attrs = {
        "year": "year",
        "month": "month",
        "week": lambda idx: idx.isocalendar().week.astype(int),
        "day_of_week": "dayofweek",
        "day_of_year": "dayofyear",
        "day_of_month": "day",
        "weekend": lambda idx: (idx.dayofweek >= 5).astype(int),
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

        if encoding is not None:
            if encoding == "onehot":
                encodable = set(_FEATURE_KNOWN_CATEGORIES.keys())
            else:  # encoding in ("cyclical", "spline")
                encodable = set(max_values.keys())
            not_encodable = [f for f in features_to_encode if f not in encodable]
            if not_encodable:
                warnings.warn(
                    f"Features {not_encodable} cannot be encoded with "
                    f"encoding={encoding!r}. Encodable features for this encoding "
                    f"are: {sorted(encodable)}. These features will be kept as "
                    f"raw integers.",
                    IgnoredArgumentWarning,
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
        effective_encode = [f for f in features_to_encode if f in _FEATURE_KNOWN_CATEGORIES]
        for feature in effective_encode:
            X_new[feature] = pd.Categorical(
                X_new[feature],
                categories=_FEATURE_KNOWN_CATEGORIES[feature],
            )
        if effective_encode:
            X_new = pd.get_dummies(
                X_new, columns=effective_encode, drop_first=False, sparse=False, dtype=int
            )
    elif encoding == "spline":
        if spline_kwargs is not None:
            blocked = {"knots", "sparse_output"}
            allowed = (
                set(inspect.signature(SplineTransformer).parameters) - blocked
            )
            invalid = set(spline_kwargs) - allowed
            if invalid:
                blocked_passed = invalid & blocked
                unknown = invalid - blocked
                msgs = []
                if blocked_passed:
                    msgs.append(
                        f"Keys {sorted(blocked_passed)} are not allowed in "
                        f"`spline_kwargs`: `knots` is computed internally from "
                        f"`max_values`, and `sparse_output` is incompatible "
                        f"with the DataFrame output."
                    )
                if unknown:
                    msgs.append(
                        f"Unknown keys in `spline_kwargs`: {sorted(unknown)}. "
                        f"Allowed keys: {sorted(allowed)}."
                    )
                raise ValueError(" ".join(msgs))

        resolved_spline_kwargs = {
            "degree": 3,
            "include_bias": True,
            "extrapolation": "periodic",
        }
        if spline_kwargs is not None:
            resolved_spline_kwargs.update(spline_kwargs)
        n_knots_global = resolved_spline_kwargs.pop("n_knots", None)
        cols_to_drop = []
        spline_cols = {}
        for feature, max_val in max_values.items():
            if feature in X_new.columns and feature in features_to_encode:
                n_knots = n_knots_global if n_knots_global is not None else max_val + 1
                min_val = _DEFAULT_MIN_VALUES.get(feature, 0)
                knots = np.linspace(min_val, max_val, n_knots).reshape(-1, 1)
                spt = SplineTransformer(
                    knots=knots,
                    **resolved_spline_kwargs,
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

    if keep_original_columns:
        if isinstance(X, pd.DataFrame):
            overlapping_cols = set(X.columns).intersection(set(X_new.columns))
            if overlapping_cols:
                raise ValueError(
                    f"The following extracted feature names already exist in the input "
                    f"DataFrame: {list(overlapping_cols)}. To avoid duplicate columns, "
                    f"rename the original columns or avoid extracting these features."
                )
            X_new = pd.concat([X, X_new], axis=1)
        else:
            if X.name in X_new.columns:
                raise ValueError(
                    f"The following extracted feature names already exist in the input "
                    f"Series: {list([X.name])}. To avoid duplicate columns, rename the "
                    f"original Series or avoid extracting these features."
                )
            X_new = pd.concat([X, X_new], axis=1)

    return X_new


class DateTimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for extracting datetime features from the DateTime index of a
    pandas DataFrame or Series. It can also apply encoding to the extracted features.

    Parameters
    ----------
    features : list, default None
        List of calendar features (strings) to extract from the index. When
        `None`, the following features are extracted: `'year'`, `'month'`,
        `'week'`, `'day_of_week'`, `'day_of_month'`, `'day_of_year'`,
        `'weekend'`, `'hour'`, `'minute'`, `'second'`, `'quarter'`. Additional
        supported features that are not extracted by default can be passed
        explicitly.
    features_to_encode : list, default None
        List of calendar features (strings) to encode. When `None`, all
        extracted features are encoded. If a feature is not in `features`, a
        `ValueError` is raised at fit/transform time. If the explicit list
        contains features that cannot be encoded with the chosen `encoding`
        (e.g. `'year'` or `'weekend'`, which are never encodable), an
        `IgnoredArgumentWarning` is issued and those features are kept as raw
        integers.
    encoding : str, default 'cyclical'
        Encoding method for the extracted features. Options are `None`,
        `'cyclical'`, `'onehot'` or `'spline'`. Features that cannot be
        encoded under the chosen mode are kept as raw integers. By default,
        `'year'` and `'weekend'` are never encoded — `'onehot'` excludes them
        via the known-category set, while `'cyclical'` and `'spline'` exclude
        them via `max_values`.
    max_values : dict, default None
        Dictionary of maximum values for the cyclical and spline encoding.
        User-provided values are **merged** with the defaults: keys passed by
        the user override the corresponding default, and missing keys fall
        back to the defaults `{'month': 12, 'week': 53, 'day_of_week': 7,
        'day_of_month': 31, 'day_of_year': 366, 'hour': 24, 'minute': 60,
        'second': 60, 'quarter': 4}`. For example, passing
        `max_values={'month': 6}` overrides only `month`; the other features
        keep their defaults. Features that are not in the defaults (e.g.
        `'year'`, `'weekend'`) are left as raw integers.
    spline_kwargs : dict, default None
        Additional keyword arguments for the spline encoding. Only used when
        `encoding='spline'`. When `None`, defaults to `{'degree': 3,
        'include_bias': True, 'extrapolation': 'periodic'}`; `n_knots`
        defaults to `max_values[feature] + 1` per feature. Knots are placed
        uniformly between the known minimum and maximum value of each feature
        (e.g. 1-12 for month, 0-23 for hour), ensuring consistent encoding
        across training and prediction. Any keyword argument accepted by
        `sklearn.preprocessing.SplineTransformer` is allowed (e.g. `n_knots`,
        `degree`, `include_bias`, `extrapolation`, `order`) **except**
        `knots` (computed internally from `max_values`) and `sparse_output`
        (incompatible with the DataFrame output). Passing either of these or
        an unknown key raises `ValueError` at fit/transform time.
    keep_original_columns : bool, default True
        If True, the original columns of `X` are kept in the output
        DataFrame. If False, only the extracted datetime features are
        returned. When `True` and `X` is an unnamed pandas Series
        (`X.name is None`), a `ValueError` is raised at fit/transform time;
        either set `X.name` to a string, or pass `keep_original_columns=False`.

    Attributes
    ----------
    features : list, None
        List of calendar features to extract from the index. `None` means the
        default features are used.
    features_to_encode : list, None
        List of calendar features to encode. `None` means all extracted
        features are encoded.
    encoding : str
        Encoding method for the extracted features.
    max_values : dict, None
        Dictionary of maximum values for the cyclical and spline encoding of
        calendar features. `None` means the default values are used.
    spline_kwargs : dict, None
        Keyword arguments for the spline encoding. `None` means the default
        values are used (`degree=3`, `include_bias=True`,
        `extrapolation='periodic'`, `n_knots=max_val+1` per feature).
    keep_original_columns : bool
        Whether to keep original columns from the input.
    feature_names_out_ : list
        Names of the output features. Set after calling `fit` or `transform`.

    Raises
    ------
    TypeError, ValueError
        Raised by `fit`, `transform`, and `fit_transform`, which delegate
        validation to `create_datetime_features`. See that function for the
        full list of conditions. Per scikit-learn convention, the constructor
        does not validate; invalid parameter values surface only when `fit`
        or `transform` is called.

    Warns
    -----
    IgnoredArgumentWarning
        See `create_datetime_features`.

    Notes
    -----
    The default `max_values` use 53 for `'week'` and 366 for `'day_of_year'`
    to accommodate the maximum possible values across all calendar years:
    ISO week 53 occurs in some years (e.g. 2015, 2020, 2026) and
    day-of-year 366 occurs in leap years. Because the encoding must be
    stateless (the same for any year, without prior knowledge of whether it
    is a leap year or contains ISO week 53), the period is fixed at the
    maximum-possible value. This implies:

    - **Onehot:** the `week_53` and `day_of_year_366` columns are always
      present in the output and equal 0 for rows whose year never reaches
      those values. This guarantees a consistent column schema across
      training and prediction.
    - **Cyclical / spline:** in years where the maximum value is reached,
      the cyclical wrap-around is exact (e.g. `sin(2π·366/366) = 0` matches
      `sin(2π·0/366) = 0`). In years where it is not, there is a one-step
      "phantom gap" between the highest observed value and 1 — the
      cyclical distance is two steps instead of one. This residual
      asymmetry is numerically small (≈ 1.7% for `day_of_year`, ≈ 12% for
      `week`) and is strictly preferable to the alternative (period
      52 / 365), which would silently collapse week 53 onto week 1 and day
      366 onto day 1 in years where those values occur.

    """

    def __init__(
        self,
        features: list[str] | None = None,
        features_to_encode: list[str] | None = None,
        encoding: str = "cyclical",
        max_values: dict[str, int] | None = None,
        spline_kwargs: dict | None = None,
        keep_original_columns: bool = True,
    ) -> None:

        self.features = features
        self.features_to_encode = features_to_encode
        self.encoding = encoding
        self.max_values = max_values
        self.spline_kwargs = spline_kwargs
        self.keep_original_columns = keep_original_columns

    def fit(self, X, y=None):
        """
        Fit the transformer by computing the output feature names.

        Parameters
        ----------
        X : pandas Series, pandas DataFrame
            Input DataFrame or Series with a datetime index.
        y : ignored
            Not used, present for API compatibility.

        Returns
        -------
        self : DateTimeFeatureTransformer
            Fitted transformer.

        """
        result = create_datetime_features(
            X=X.iloc[:2] if isinstance(X, (pd.DataFrame, pd.Series)) else X,
            features=self.features,
            features_to_encode=self.features_to_encode,
            encoding=self.encoding,
            max_values=self.max_values,
            spline_kwargs=self.spline_kwargs,
            keep_original_columns=self.keep_original_columns,
        )
        self.feature_names_out_ = list(result.columns)

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
                    X                     = X,
                    encoding              = self.encoding,
                    features              = self.features,
                    features_to_encode    = self.features_to_encode,
                    max_values            = self.max_values,
                    spline_kwargs         = self.spline_kwargs,
                    keep_original_columns = self.keep_original_columns,
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


def _freq_to_timedelta_unit(freq_str: str) -> str:
    """
    Map a pandas frequency string to a numpy timedelta unit.

    Coarser-than-hourly frequencies (daily, weekly, monthly, etc.) always map to
    days because holiday dates are calendar dates and sub-daily distances to them
    are always expressed as whole days.

    Parameters
    ----------
    freq_str : str
        Pandas frequency string (e.g. `'D'`, `'h'`, `'2min'`, `'W-MON'`).

    Returns
    -------
    unit : str
        Numpy timedelta unit string (e.g. `'D'`, `'h'`, `'m'`).

    """
    # Strip leading digit multiplier and weekday suffix (e.g. '2h' -> 'h', 'W-MON' -> 'W')
    normalized = re.split(r'[-_]', freq_str)[0]
    normalized = re.sub(r'^\d+', '', normalized)

    _coarse = {
        'YE', 'YS', 'Y', 'A', 'AS', 'QE', 'QS', 'Q',
        'ME', 'MS', 'M', 'BM', 'BMS', 'BME', 'CBM', 'CBMS', 'CBME',
        'W', 'D', 'B', 'C',
    }
    if normalized in _coarse:
        unit = 'D'
    elif normalized in {'h', 'H'}:
        unit = 'h'
    elif normalized in {'min', 'T'}:
        unit = 'm'
    elif normalized in {'s', 'S'}:
        unit = 's'
    elif normalized in {'ms', 'L'}:
        unit = 'ms'
    elif normalized in {'us', 'U'}:
        unit = 'us'
    elif normalized in {'ns', 'N'}:
        unit = 'ns'
    else:
        unit = 'h'  # default to hours if frequency is unknown

    return unit


def calculate_distance_from_holiday(
    X: pd.DataFrame | pd.Series,
    holiday_column: str | None = None,
    date_column: str | None = None,
    fill_na: int | float = 0.,
) -> pd.DataFrame:
    """
    Calculate the number of periods to the next and since the last holiday.

    The time unit used for the calculation (days, hours, minutes, …) is inferred
    from the frequency of the index when `date_column=None`, and is always days
    when a date column is used instead. Output columns are always named
    `time_to_holiday` and `time_since_holiday` regardless of the unit.

    Parameters
    ----------
    X : pandas Series, pandas DataFrame
        Input data containing the holiday indicator. When a Series is passed,
        its values are used directly as the holiday indicator (boolean or 0/1)
        and `holiday_column` is ignored. When a DataFrame is passed,
        `holiday_column` must be specified.
    holiday_column : str, None, default None
        Name of the boolean column indicating holidays (`True` or `1` on holiday
        dates, `False` or `0` otherwise). Required when `X` is a pandas
        DataFrame. Ignored when `X` is a pandas Series.
    date_column : str, None, default None
        Name of the column containing dates to use as reference. When `None`,
        the index is used and must be a pandas DatetimeIndex.
    fill_na : int, float, default 0.
        Value used to fill rows where no previous or next holiday exists (i.e.
        before the first holiday or after the last). Pass `numpy.nan` to keep
        those entries as `NaN`.

    Returns
    -------
    result : pandas DataFrame
        DataFrame with two new columns and the same index as `X`:

        - `time_to_holiday`: periods until the next holiday.
        - `time_since_holiday`: periods since the last holiday.

    Notes
    -----
    When `date_column` is specified, the unit is always days regardless of the
    data frequency, because no index frequency information is available.

    When `date_column=None`, the time unit is inferred from the index frequency:

    - Daily or coarser (weekly, monthly, …): days
    - Hourly: hours
    - Minute: minutes
    - Second: seconds
    - Millisecond: milliseconds
    - Microsecond: microseconds
    - Nanosecond: nanoseconds

    When the index has no frequency set, `pd.infer_freq` is attempted. If the
    frequency still cannot be determined (e.g. irregular spacing or fewer than
    three observations), the unit defaults to hours and a `UserWarning` is
    issued.

    """
    if not isinstance(X, (pd.DataFrame, pd.Series)):
        raise TypeError(
            "Input `X` must be a pandas Series or pandas DataFrame."
        )

    if isinstance(X, pd.Series):
        if holiday_column is not None:
            warnings.warn(
                "`holiday_column` is ignored when `X` is a pandas Series. "
                "The Series values are used directly as the holiday indicator.",
                IgnoredArgumentWarning,
                stacklevel=2,
            )
        col_name = X.name if X.name is not None else "is_holiday"
        X = X.rename(col_name).to_frame()
        holiday_column = col_name
    else:
        if holiday_column is None:
            raise ValueError(
                "`holiday_column` must be specified when `X` is a pandas DataFrame."
            )

    if date_column is None:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError(
                "When `date_column=None`, the index must be a pandas DatetimeIndex."
            )
        dates = X.index.to_numpy()
        holiday_dates = X.index[X[holiday_column].astype(bool)].to_numpy()

        freq_str = X.index.freqstr if X.index.freq is not None else None
        if freq_str is None:
            freq_str = pd.infer_freq(X.index)
        if freq_str is None:
            warnings.warn(
                "Could not determine the frequency of the index. "
                "The output column unit defaults to 'hours'. Set the index "
                "frequency with `X.asfreq(...)` to avoid this warning.",
                UserWarning,
                stacklevel=2,
            )
            freq_str = 'h'
        unit = _freq_to_timedelta_unit(freq_str)
    else:
        dates = pd.to_datetime(X[date_column]).to_numpy()
        holiday_dates = pd.to_datetime(
            X.loc[X[holiday_column].astype(bool), date_column]
        ).to_numpy()
        unit = 'D'

    holiday_dates_sorted = np.sort(holiday_dates)

    # Periods until the next holiday
    next_idx = np.searchsorted(holiday_dates_sorted, dates, side='left')
    has_next = next_idx < len(holiday_dates_sorted)
    to_holiday = np.full(len(dates), np.nan)
    to_holiday[has_next] = (
        holiday_dates_sorted[next_idx[has_next]] - dates[has_next]
    ).astype(f'timedelta64[{unit}]').astype(int)

    # Periods since the last holiday
    prev_idx = np.searchsorted(holiday_dates_sorted, dates, side='right') - 1
    has_prev = prev_idx >= 0
    since_holiday = np.full(len(dates), np.nan)
    since_holiday[has_prev] = (
        dates[has_prev] - holiday_dates_sorted[prev_idx[has_prev]]
    ).astype(f'timedelta64[{unit}]').astype(int)

    to_col = pd.Series(to_holiday, index=X.index, dtype="Int64").fillna(fill_na)
    since_col = pd.Series(since_holiday, index=X.index, dtype="Int64").fillna(fill_na)

    return pd.DataFrame(
        {"time_to_holiday": to_col, "time_since_holiday": since_col}
    )
