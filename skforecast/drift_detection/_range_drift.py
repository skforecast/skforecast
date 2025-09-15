################################################################################
#                             RangeDriftDetector                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import pandas as pd
import numpy as np
import warnings
import textwrap
from rich.console import Console
from rich.panel import Panel
from ..exceptions import (
    FeatureOutOfRangeWarning,
    IgnoredArgumentWarning,
    MissingExogWarning,
    UnknownLevelWarning
)
from ..utils import set_skforecast_warnings

class RangeDriftDetector:
    """
    Detector of out-of-range values based on training feature ranges.

    The detector is intentionally lightweight: it does not compute advanced
    drift statistics since it is used to check single observations during
    inference. Suitable for real-time applications.

    Parameters
    ----------

    Attributes
    ----------
    series_values_range_ : dict
        Range of values of the target series used during training.
    exog_values_range_ : dict
        Range of values of the exogenous variables used during training.
    series_names_in_ : list
        Names of the series used during training.
    exog_names_in_ : list
        Names of the exogenous variables used during training.
    is_fitted_ : bool
        Whether the detector has been fitted to the training data.
    """

    def __init__(self):
        self.series_names_in_     = None
        self.series_values_range_ = None
        self.exog_names_in_       = None
        self.exog_values_range_   = None
        self.is_fitted_           = False

    def __repr__(self) -> str:
        """
        Information displayed when a RangeDriftDetector object is printed.
        """

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Series value ranges: {self.series_values_range_} \n"
            f"Exogenous value ranges: {self.exog_values_range_} \n"
            f"Fitted series: {self.series_names_in_} \n"
            f"Fitted exogenous: {self.exog_names_in_} \n"
        )

        return info

    @classmethod
    def _get_features_range(cls, X: pd.DataFrame | pd.Series) -> tuple | set | dict:
        """
        Get a summary of the features in the DataFrame or Series. For numeric
        features, it returns the min and max values. For categorical features,
        it returns the unique values.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series
            Input data to summarize.
        Returns
        -------
        features_ranges: tuple, set, dict
            Feature ranges. If X is a Series, returns a tuple (min, max) for numeric
            data or a set of unique values for categorical data. If X is a DataFrame,
            returns a dictionary with column names as keys and their respective ranges
            (tuple or set) as values.

        """

        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise TypeError("Input must be a pandas DataFrame or Series.")

        if isinstance(X, pd.Series):
            if pd.api.types.is_numeric_dtype(X):
                features_ranges = (X.min(), X.max())
            else:
                features_ranges = set(X.dropna().unique())

        if isinstance(X, pd.DataFrame):
            num_cols = [
                col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
            ]
            cat_cols = [col for col in X.columns if col not in num_cols]

            features_ranges = {}
            features_ranges.update(
                {col: (X[col].min(), X[col].max()) for col in num_cols}
            )
            features_ranges.update(
                {col: set(X[col].dropna().unique()) for col in cat_cols}
            )

        return features_ranges

    @classmethod
    def _check_feature_range(
        cls,
        feature_range: tuple | set,
        X: pd.Series
    ) -> bool:
        """
        Check if there is any value outside the training range. For numeric features,
        it checks if the values are within the min and max range. For categorical features,
        it checks if the values are among the seen categories.

        Parameters
        ----------
        feature_range : tuple, set
            Output from _get_features_range() for a single feature.
        X : pd.Series
            New data to validate

        Returns
        -------
        bool
            True if there is any value outside the training range, False otherwise.
        """

        if isinstance(feature_range, tuple):
            return X.min() < feature_range[0] or X.max() > feature_range[1]
        else:
            unseen = set(X.dropna().unique()) - feature_range
            return bool(unseen)

    @classmethod
    def _display_warnings(
        cls,
        not_compliant_feature: str,
        feature_range: tuple | set,
        series_name: str = None,
    ) -> None:
        """
        Display warnings for features with values outside the training range.

        Parameters
        ----------
        not_compliant_feature : str
            Name of the feature with values outside the training range.
        feature_range : tuple | set
            Training range of the feature.
        series_name : str, optional
            Name of the series being checked, if applicable.

        Returns
        -------
        None

        """
        if isinstance(feature_range, tuple):  # numeric
            msg = (
                f"'{not_compliant_feature}' has values outside the range seen during training "
                f"[{feature_range[0]:.5f}, {feature_range[1]:.5f}]. "
                f"This may affect the accuracy of the predictions."
            )
        else:  # categorical
            msg = (
                f"'{not_compliant_feature}' has values not seen during training. Seen values: "
                f"{feature_range}. This may affect the accuracy of the predictions."
            )

        if series_name:
            msg = f"'{series_name}': " + msg

        warnings.warn(msg, FeatureOutOfRangeWarning)

    @classmethod
    def _summary(
        cls,
        out_of_range_series: list,
        out_of_range_series_ranges: list,
        out_of_range_exog: list,
        out_of_range_exog_ranges: list,
        out_of_range_exog_series_id: list,
    ):
        """
        Summarize the results of the range check.

        Parameters
        ----------
        out_of_range_series : list
            List of series names that are out of range.
        out_of_range_series_ranges : list
            List of ranges for the out-of-range series.
        out_of_range_exog : list
            List of exogenous variable names that are out of range.
        out_of_range_exog_ranges : list
            List of ranges for the out-of-range exogenous variables.
        out_of_range_exog_series_id : list
            List of series IDs for the out-of-range exogenous variables. This is
            used when exogenous variables are different for each series.
        """
        
        msg_series = ""
        msg_exog = ""
        if out_of_range_series:
            series_msgs = []
            for series, series_range in zip(
                out_of_range_series, out_of_range_series_ranges
            ):
                msg_temp = (
                    f"'{series}' has values outside the observed range "
                    f"[{series_range[0]:.5f}, {series_range[1]:.5f}]."
                )
                series_msgs.append(textwrap.fill(msg_temp, width=80))
            msg_series = "\n".join(series_msgs) + "\n"
        else:
            msg_series = "No series with out-of-range values found.\n"

        if out_of_range_exog:
            exog_msgs = []
            for exog, exog_range, series_id in zip(
                out_of_range_exog, out_of_range_exog_ranges, out_of_range_exog_series_id
            ):
                if isinstance(exog_range, tuple):  # numeric
                    msg_temp = (
                        f"'{exog}' has values outside the observed range "
                        f"[{exog_range[0]:.5f}, {exog_range[1]:.5f}]."
                    )
                else:  # categorical
                    msg_temp = (
                        f"'{exog}' has values not seen during training. Seen values: "
                        f"{exog_range}."
                    )
                if series_id:
                    msg_temp = f"'{series_id}': " + msg_temp
                exog_msgs.append(textwrap.fill(msg_temp, width=80))
            msg_exog = "\n".join(exog_msgs) + "\n"
        else:
            msg_exog = "No exogenous variables with out-of-range values found.\n"

        console = Console()
        content = (
            f"[bold]Series:[/bold]\n{msg_series}\n"
            f"[bold]Exogenous Variables:[/bold]\n{msg_exog}\n"
        )
        console.print(Panel(content, title=f"[bold]Out-of-range summary[/bold]", expand=False))

    @classmethod
    def _normalize_input(cls, X, name: str) -> dict:
        """
        Convert pd.Series, pd.DataFrame or dict into a standardized dict of
        pd.Series or pd.DataFrames.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, dict
            Input data to normalize.
        name : str
            Name of the input being normalized. Used for error messages.
            Expected values are 'series', 'last_window' or 'exog'.
        """
        if isinstance(X, pd.Series):
            if not X.name:
                raise ValueError(
                    f"{name} must have a name when a pandas Series is provided."
                )
            X = {X.name: X}

        elif isinstance(X, pd.DataFrame):
            if isinstance(X.index, pd.MultiIndex):
                if name in ["series", "last_window"]:
                    col = X.columns[0]
                    if len(X.columns) != 1:
                        warnings.warn(
                            f"`{name}` DataFrame has multiple columns. Only the first column "
                            f"'{col}' will be used. Others ignored.",
                            IgnoredArgumentWarning,
                        )
                    X = {
                        series_id: X.loc[series_id][col].rename(series_id)
                        for series_id in X.index.levels[0]
                    }
                else:
                    X = {series_id: X.loc[series_id] for series_id in X.index.levels[0]}
            else:
                X = X.to_dict(orient="series")

        elif isinstance(X, dict):
            for k, v in X.items():
                if not isinstance(v, (pd.Series, pd.DataFrame)):
                    raise TypeError(
                        f"All values in `{name}` must be Series or DataFrame."
                    )

        return X
    
    def fit(
        self,
        series: pd.DataFrame | pd.Series | dict | None = None,
        exog: pd.DataFrame | pd.Series | dict | None = None,
        **kwargs
    ) -> None:
        """
        Fit detector, storing training ranges.

        Parameters
        ----------
        series : pd.DataFrame, pd.Series, dict
            Input time series data to fit the detector, ideally the same ones
            used to fit the forecaster.
        exog : pd.DataFrame, pd.Series, dict | None, optional
            Exogenous variables to include in the forecaster.

        Returns
        -------
        None

        """

        # Deprecation of 'y' argument in favor of 'series'
        if series is None and 'y' not in kwargs:
            raise ValueError(
                "`series` cannot be None. Please provide the time series data."
            )
        if 'y' in kwargs:
            if series is not None:
                raise TypeError(
                    "Cannot specify both 'series' and 'y'. Please use 'series' "
                    "since 'y' is deprecated."
                )
            else:
                warnings.warn(
                    "`y` is deprecated and will be removed in a future version. "
                    "Please use 'series' instead.",
                    FutureWarning,
                )
                series = kwargs.pop('y')
        
        self.series_values_range_ = {}
        self.series_names_in_     = []
        self.exog_values_range_   = None
        self.exog_names_in_       = None

        if not isinstance(series, (pd.DataFrame, pd.Series, dict)):
            raise TypeError("Input must be a pandas DataFrame, Series or dict.")

        if not isinstance(exog, (pd.DataFrame, pd.Series, dict, type(None))):
            raise TypeError(
                "Exogenous variables must be a pandas DataFrame, Series or dict."
            )

        series = self._normalize_input(series, name="series")
        for key, value in series.items():
            self.series_values_range_[key] = self._get_features_range(X=value)
            self.series_names_in_.append(key)

        if exog is not None:

            exog = self._normalize_input(exog, name="exog")

            self.exog_values_range_ = {}
            self.exog_names_in_ = []

            for key, value in exog.items():
                self.exog_values_range_[key] = self._get_features_range(X=value)
                if isinstance(value, pd.Series):
                    self.exog_names_in_.append(key)
                else:
                    self.exog_names_in_.extend(value.columns)

            self.exog_names_in_ = list(dict.fromkeys(self.exog_names_in_))

        self.is_fitted_ = True

        return

    def predict(
        self,
        last_window: pd.Series | pd.DataFrame | dict | None = None,
        exog: pd.Series | pd.DataFrame | dict | None = None,
        verbose: bool = True,
        suppress_warnings: bool = False
    ) -> tuple[bool, list, list]:
        """
        Check if there is any value outside the training range for last_window and exog.

        Parameters
        ----------
        last_window : pandas Series, pandas DataFrame, dict, default None
            Series values used to create the predictors (lags) needed in the
            first iteration of the prediction (t + 1).
        exog : pandas Series, pandas DataFrame, dict, default None
            Exogenous variable/s included as predictor/s.
        verbose : bool, default False
            Whether to print a summary of the check.
        suppress_warnings : bool, default False
            Whether to suppress warnings.

        Returns
        -------
        flag_out_of_range : bool
            True if there is any value outside the training range, False otherwise.
        out_of_range_series : list
            List of series names that are out of range.
        out_of_range_exog : list
            List of exogenous variable names that are out of range.

        """

        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet.")

        if not isinstance(last_window, (pd.DataFrame, pd.Series, dict, type(None))):
            raise TypeError(
                "last_window must be a pandas DataFrame, Series, dict or None."
            )

        if not isinstance(exog, (pd.DataFrame, pd.Series, dict, type(None))):
            raise TypeError(
                "Exogenous variables must be a pandas DataFrame, Series, dict or None."
            )
        
        set_skforecast_warnings(suppress_warnings, action='ignore')
        
        flag_out_of_range = False
        out_of_range_series = []
        out_of_range_series_ranges = []
        if last_window is not None:
            last_window = self._normalize_input(last_window, name="last_window")
            for key, value in last_window.items():
                if isinstance(value, pd.Series):
                    value = value.to_frame()
                for col in value.columns:
                    if key not in self.series_names_in_:
                        warnings.warn(
                            f"'{key}' was not seen during training. Its range is unknown.",
                            UnknownLevelWarning,
                        )
                        continue
                    is_out_of_range = self._check_feature_range(
                        feature_range=self.series_values_range_[col], X=value[col]
                    )
                    if is_out_of_range:
                        flag_out_of_range = True
                        out_of_range_series.append(col)
                        out_of_range_series_ranges.append(self.series_values_range_[col])
                        self._display_warnings(
                            not_compliant_feature=col,
                            feature_range=self.series_values_range_[col],
                            series_name=None,
                        )

        out_of_range_exog_series_id = []
        out_of_range_exog = []
        out_of_range_exog_ranges = []
        if exog is not None:
            exog = self._normalize_input(exog, name="exog")
            for key, value in exog.items():
                if isinstance(value, pd.Series):
                    value = value.to_frame()
                features_ranges = self.exog_values_range_.get(key, None)
                for col in value.columns:
                    if not isinstance(features_ranges, dict):
                        is_single_series = True
                        features_ranges = {key: features_ranges}
                    else:
                        is_single_series = False
                    if col not in self.exog_names_in_:
                        warnings.warn(
                            f"'{col}' was not seen during training. Its range is unknown.",
                            MissingExogWarning,
                        )
                        continue
                    is_out_of_range = self._check_feature_range(
                        feature_range=features_ranges[col], X=value[col]
                    )
                    if is_out_of_range:
                        flag_out_of_range = True
                        out_of_range_exog.append(col)
                        out_of_range_exog_ranges.append(features_ranges[col])
                        out_of_range_exog_series_id.append(key if not is_single_series else None)
                        self._display_warnings(
                            not_compliant_feature=col,
                            feature_range=features_ranges[col],
                            series_name=key if not is_single_series else None,
                        )

        if verbose:
            self._summary(
                out_of_range_series,
                out_of_range_series_ranges,
                out_of_range_exog,
                out_of_range_exog_ranges,
                out_of_range_exog_series_id
            )

        set_skforecast_warnings(suppress_warnings, action='default')

        return flag_out_of_range, out_of_range_series, out_of_range_exog