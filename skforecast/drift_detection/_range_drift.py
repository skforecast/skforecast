################################################################################
#                             RangeDriftDetector                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import numpy as np
import pandas as pd
import warnings
import textwrap
from rich.console import Console
from rich.panel import Panel
import skforecast
from ..exceptions import (
    FeatureOutOfRangeWarning,
    IgnoredArgumentWarning,
    MissingExogWarning,
    UnknownLevelWarning
)
from ..utils import (
    set_skforecast_warnings,
    get_style_repr_html
)


class RangeDriftDetector:
    """
    Detector of out-of-range values based on training feature ranges.

    The detector is intentionally lightweight: it does not compute advanced
    drift statistics since it is used to check single observations during
    inference. Suitable for real-time applications.

    Parameters
    ----------
    None

    Attributes
    ----------
    series_names_in_ : list
        Names of the series used during training.
    series_values_range_ : dict
        Range of values of the target series used during training.
    exog_names_in_ : list
        Names of the exogenous variables used during training.
    exog_values_range_ : dict
        Range of values of the exogenous variables used during training.
    series_specific_exog_ : bool
        Indicates whether exogenous variables have different values across
        target series during training (i.e., exogenous is series-specific
        rather than global).
    is_fitted : bool
        Whether the detector has been fitted to the training data.
    
    """

    def __init__(self) -> None:

        self.series_names_in_      = None
        self.series_values_range_  = None
        self.exog_names_in_        = None
        self.exog_values_range_    = None
        self.series_specific_exog_ = False
        self.is_fitted             = False

    def __repr__(self) -> str:
        """
        Information displayed when a RangeDriftDetector object is printed.
        """
    
        series_names_in_ = None
        if self.series_names_in_ is not None:
            if len(self.series_names_in_) > 50:
                series_names_in_ = self.series_names_in_[:25] + ["..."] + self.series_names_in_[-25:]
            series_names_in_ = ", ".join(self.series_names_in_)

        exog_names_in_ = None
        if self.exog_names_in_ is not None:
            if len(self.exog_names_in_) > 50:
                exog_names_in_ = self.exog_names_in_[:25] + ["..."] + self.exog_names_in_[-25:]
            exog_names_in_ = ", ".join(self.exog_names_in_)

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Fitted series          = {series_names_in_} \n"
            f"Series value ranges    = {self.series_values_range_} \n"
            f"Fitted exogenous       = {exog_names_in_} \n"
            f"Exogenous value ranges = {self.exog_values_range_} \n"
            f"Series-specific exog   = {self.series_specific_exog_} \n"
            f"Is fitted              = {self.is_fitted}"
        )

        return info

    def _repr_html_(self):
        """
        HTML representation of the object.
        The "General Information" section is expanded by default.
        """
    
        series_names_in_ = None
        if self.series_names_in_ is not None:
            if len(self.series_names_in_) > 50:
                series_names_in_ = self.series_names_in_[:25] + ["..."] + self.series_names_in_[-25:]
            series_names_in_ = ", ".join(self.series_names_in_)

        exog_names_in_ = None
        if self.exog_names_in_ is not None:
            if len(self.exog_names_in_) > 50:
                exog_names_in_ = self.exog_names_in_[:25] + ["..."] + self.exog_names_in_[-25:]
            exog_names_in_ = ", ".join(self.exog_names_in_)

        style, unique_id = get_style_repr_html(self.is_fitted)
        content = f"""
        <div class="container-{unique_id}">
            <h2>{type(self).__name__}</h2>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Fitted series:</strong> {series_names_in_}</li>
                    <li><strong>Fitted exogenous:</strong> {exog_names_in_}</li>
                    <li><strong>Series-specific exogenous:</strong> {self.series_specific_exog_}</li>
                    <li><strong>Is fitted:</strong> {self.is_fitted}</li>
                </ul>
            </details>
            <details>
                <summary>Series value ranges</summary>
                <ul>
                    {self.series_values_range_}
                </ul>
            </details>
            <details>
                <summary>Exogenous value ranges</summary>
                <ul>
                    {self.exog_values_range_}
                </ul>
            </details>
            <p>
                <a href="https://skforecast.org/{skforecast.__version__}/api/drift_detection.html#skforecast.drift_detection.drift_detection.RangeDriftDetector">&#128712 <strong>API Reference</strong></a>
                &nbsp;&nbsp;
                <a href="https://skforecast.org/{skforecast.__version__}/user_guides/drift-detection.html">&#128462 <strong>User Guide</strong></a>
            </p>
        </div>
        """
        
        return style + content

    @classmethod
    def _get_features_range(
        cls, 
        X: pd.Series | pd.DataFrame
    ) -> tuple | set | dict[str, tuple | set]:
        """
        Get a summary of the features in the DataFrame or Series. For numeric
        features, it returns the min and max values. For categorical features,
        it returns the unique values.

        Parameters
        ----------
        X : pandas Series, pandas DataFrame
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
                features_ranges = (float(X.min()), float(X.max()))
            else:
                features_ranges = set(X.dropna().unique())

        if isinstance(X, pd.DataFrame):
            num_cols = [
                col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
            ]
            cat_cols = [col for col in X.columns if col not in num_cols]

            features_ranges = {}
            features_ranges.update(
                {col: (float(X[col].min()), float(X[col].max())) for col in num_cols}
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

        if isinstance(feature_range, tuple):
            # Numeric
            msg = (
                f"'{not_compliant_feature}' has values outside the range seen during training "
                f"[{feature_range[0]:.5f}, {feature_range[1]:.5f}]. "
                f"This may affect the accuracy of the predictions."
            )
        else:
            # Categorical
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
        out_of_range_exog_ranges: list
    ) -> None:
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

        Returns
        -------
        None

        """
        
        msg_series = ""
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

        msg_exog = ""
        if out_of_range_exog:
            exog_msgs = []
            if isinstance(out_of_range_exog, list):
                for exog, exog_range in zip(out_of_range_exog, out_of_range_exog_ranges):
                    if isinstance(exog_range, tuple):
                        # Numeric
                        msg_temp = (
                            f"'{exog}' has values outside the observed range "
                            f"[{exog_range[0]:.5f}, {exog_range[1]:.5f}]."
                        )
                    else:
                        # Categorical
                        msg_temp = (
                            f"'{exog}' has values not seen during training. Seen values: "
                            f"{exog_range}."
                        )
                    exog_msgs.append(textwrap.fill(msg_temp, width=80))
            else:
                for key, value in out_of_range_exog.items():
                    for exog, exog_range in zip(value, out_of_range_exog_ranges[key]):
                        if isinstance(exog_range, tuple):
                            # Numeric
                            msg_temp = (
                                f"'{exog}' has values outside the observed range "
                                f"[{exog_range[0]:.5f}, {exog_range[1]:.5f}]."
                            )
                        else:
                            # Categorical
                            msg_temp = (
                                f"'{exog}' has values not seen during training. Seen values: "
                                f"{exog_range}."
                            )
                        msg_temp = f"'{key}': " + msg_temp
                        exog_msgs.append(textwrap.fill(msg_temp, width=80))

            msg_exog = "\n".join(exog_msgs)
        else:
            msg_exog = "No exogenous variables with out-of-range values found."

        console = Console()
        content = (
            f"[bold]Series:[/bold]\n{msg_series}\n"
            f"[bold]Exogenous Variables:[/bold]\n{msg_exog}"
        )
        console.print(Panel(content, title="[bold]Out-of-range summary[/bold]", expand=False))

    def _normalize_input(
        self, 
        X: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
        name: str,
        series_ids: list[str] | None = None
    ) -> dict[str, pd.Series | pd.DataFrame]:
        """
        Convert pd.Series, pd.DataFrame or dict into a standardized dict of
        pd.Series or pd.DataFrames.

        Parameters
        ----------
        X : pandas Series, pandas DataFrame, dict
            Input data to normalize.
        name : str
            Name of the input being normalized. Used for error messages.
            Expected values are 'series', 'last_window' or 'exog'.
        series_ids : list, default None
            Series IDs to include in the normalization of exogenous variables.

        Returns
        -------
        X : dict
            Normalized input as a dictionary of pandas Series or DataFrames.

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
                            f"`{name}` DataFrame has multiple columns. Only the "
                            f"first column, '{col}', will be used. Others ignored.",
                            IgnoredArgumentWarning,
                        )
                    X = {
                        series_id: X.loc[series_id][col].rename(series_id)
                        for series_id in X.index.levels[0]
                    }
                else:
                    X = {series_id: X.loc[series_id] for series_id in X.index.levels[0]}
            else:
                if self.series_specific_exog_ and series_ids:
                    X = {series_id: X.copy() for series_id in series_ids}
                else:
                    X = X.to_dict(orient="series")

        elif isinstance(X, dict):
            for k, v in X.items():
                if not isinstance(v, (pd.Series, pd.DataFrame)):
                    raise TypeError(
                        f"All values in `{name}` must be a pandas Series or DataFrame. "
                        f"Review the value for key '{k}'."
                    )

        return X
    
    def fit(
        self,
        series: pd.DataFrame | pd.Series | dict[str, pd.Series | pd.DataFrame] | None = None,
        exog: pd.DataFrame | pd.Series | dict[str, pd.Series | pd.DataFrame] | None = None,
        **kwargs
    ) -> None:
        """
        Fit detector, storing training ranges.

        Parameters
        ----------
        series : pandas Series, pandas DataFrame, dict, aliases: `y`
            Input time series data to fit the detector, ideally the same ones
            used to fit the forecaster.
        exog : pandas Series, pandas DataFrame, dict, default None
            Exogenous variables to include in the forecaster.

        Returns
        -------
        None

        """

        if series is None and ('y' not in kwargs or kwargs['y'] is None):
            raise ValueError(
                "One of `series` or `y` must be provided."
            )
        if 'y' in kwargs:
            if series is not None:
                raise ValueError(
                    "Cannot specify both `series` and `y`. Please provide only one of them."
                )
            series = kwargs.pop('y')

        if not isinstance(series, (pd.Series, pd.DataFrame, dict)):
            raise TypeError("Input must be a pandas Series, DataFrame or dict.")

        if not isinstance(exog, (pd.Series, pd.DataFrame, dict, type(None))):
            raise TypeError(
                "Exogenous variables must be a pandas Series, DataFrame or dict."
            )
        
        self.series_names_in_      = []
        self.series_values_range_  = {}
        self.exog_names_in_        = None
        self.exog_values_range_    = None
        self.series_specific_exog_ = False
        self.is_fitted             = False

        series = self._normalize_input(series, name="series")
        for key, value in series.items():
            self.series_names_in_.append(key)
            self.series_values_range_[key] = self._get_features_range(X=value)

        if exog is not None:

            exog = self._normalize_input(exog, name="exog")

            self.exog_names_in_ = []
            self.exog_values_range_ = {}
            for key, value in exog.items():
                if isinstance(value, pd.Series):
                    self.exog_names_in_.append(key)
                else:
                    self.exog_names_in_.extend(value.columns)
                self.exog_values_range_[key] = self._get_features_range(X=value)

            self.exog_names_in_ = list(dict.fromkeys(self.exog_names_in_))
            self.series_specific_exog_ = any(key in self.series_names_in_ for key in exog.keys())

        self.is_fitted = True

    def predict(
        self,
        last_window: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
        exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
        verbose: bool = True,
        suppress_warnings: bool = False
    ) -> tuple[bool, list[str], list[str] | dict[str, list[str]]]:
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
        out_of_range_exog : list, dict
            Exogenous variables that are out of range.

            - If `self.series_specific_exog_` is False: returns a list with the names
            of exogenous variables that are out of range (global exogenous).
            - If `self.series_specific_exog_` is True: returns a dictionary where
            keys are series names and values are lists of out-of-range exogenous
            variables for each series.

        """

        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if not isinstance(last_window, (pd.Series, pd.DataFrame, dict, type(None))):
            raise TypeError(
                "`last_window` must be a pandas Series, DataFrame, dict or None."
            )

        if not isinstance(exog, (pd.Series, pd.DataFrame, dict, type(None))):
            raise TypeError(
                "`exog` must be a pandas Series, DataFrame, dict or None."
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
                            not_compliant_feature = col,
                            feature_range         = self.series_values_range_[col],
                            series_name           = None
                        )

        out_of_range_exog = {} if self.series_specific_exog_ else []
        out_of_range_exog_ranges = {} if self.series_specific_exog_ else []
        if exog is not None:
            series_ids = list(last_window.keys()) if last_window is not None else self.series_names_in_
            exog = self._normalize_input(exog, name="exog", series_ids=series_ids)
            for key, value in exog.items():

                if isinstance(value, pd.Series):
                    value = value.to_frame()
                features_ranges = self.exog_values_range_.get(key, None)
                
                if self.series_specific_exog_:
                    out_of_range_exog[key] = []
                    out_of_range_exog_ranges[key] = []
                
                for col in value.columns:

                    if not isinstance(features_ranges, dict):
                        features_ranges = {key: features_ranges}
                    
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
                        if self.series_specific_exog_:
                            out_of_range_exog[key].append(col)
                            out_of_range_exog_ranges[key].append(features_ranges[col])
                        else:
                            out_of_range_exog.append(col)
                            out_of_range_exog_ranges.append(features_ranges[col])

                        self._display_warnings(
                            not_compliant_feature = col,
                            feature_range         = features_ranges[col],
                            series_name           = key if self.series_specific_exog_ else None,
                        )

                if self.series_specific_exog_ and not out_of_range_exog[key]:
                    out_of_range_exog.pop(key)
                    out_of_range_exog_ranges.pop(key)

        if verbose:
            self._summary(
                out_of_range_series        = out_of_range_series,
                out_of_range_series_ranges = out_of_range_series_ranges,
                out_of_range_exog          = out_of_range_exog,
                out_of_range_exog_ranges   = out_of_range_exog_ranges
            )

        set_skforecast_warnings(suppress_warnings, action='default')

        return flag_out_of_range, out_of_range_series, out_of_range_exog
