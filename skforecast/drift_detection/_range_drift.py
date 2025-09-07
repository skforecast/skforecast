################################################################################
#                             RangeDriftDetector                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import pandas as pd
import numpy as np
import warnings
from ..exceptions import FeatureOutOfRangeWarning, IgnoredArgumentWarning


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
    series_type_in_ : type
        Type of series data (pandas Series, DataFrame or dict) used in training.
    exog_type_in_ : type
        Type of exogenous data (pandas Series, DataFrame or dict) used in training.
    """

    def __init__(self):
        self.series_names_in_ = []
        self.series_values_range_ = {}
        self.exog_names_in_ = []
        self.exog_values_range_ = {}
        self.is_fitted_ = False
        self.series_dtypes_in_ = None
        self.exog_dtypes_in_ = None

    def __repr__(
        self
    ) -> str:
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
            f"Series data type: {self.series_dtypes_in_} \n"
            f"Exogenous data type: {self.exog_dtypes_in_} \n"
        )

        return info

    @classmethod
    def _get_features_range(cls, X: pd.DataFrame | pd.Series) -> tuple | set | dict:
        """
        Get a summary of the features in the DataFrame or Series. For numeric
        features, it returns the min and max values. For categorical features,
        it returns the unique values.

        Arguments
        ---------
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
            num_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
            cat_cols = [col for col in X.columns if col not in num_cols]

            features_ranges = {}
            features_ranges.update({col: (X[col].min(), X[col].max()) for col in num_cols})
            features_ranges.update({col: set(X[col].dropna().unique()) for col in cat_cols})

        return features_ranges

    @classmethod
    def check_features_range_(
        cls,
        features_ranges: dict, 
        X: pd.DataFrame | pd.Series,
    ) -> list[str]:
        """
        Check if there is any value outside the training range. For numeric features,
        it checks if the values are within the min and max range. For categorical features,
        it checks if the values are among the seen categories.

        Parameters
        ----------
        features_ranges : dict
            Output from _get_features_range()
        X : pd.DataFrame, pd.Series
            New data to validate

        Returns
        -------
        not_compliant_features : list[str]
            List of features with values outside the training range.
        """

        if isinstance(X, pd.Series):
            X = X.to_frame()

        for col in set(X.columns).intersection(features_ranges.keys()):
            rule = features_ranges[col]

        not_compliant_features = []
        if isinstance(rule, tuple):
            if X[col].min() < rule[0] or X[col].max() > rule[1]:
                not_compliant_features.append(col)
        else:
            unseen = set(X[col].unique()) - rule
            if unseen:
                not_compliant_features.append(col)
        
        return not_compliant_features
                
    @classmethod
    def _display_warnings(
        cls, 
        not_compliant_features: list[str],
        feature_values_range: dict,
        series_name: str = None
    ) -> None:
        """
        Display warnings for features with values outside the training range.

        Parameters
        ----------
        not_compliant_features : list[str]
            List of feature names with values outside the training range.
        feature_values_range : dict
            Dictionary with the training ranges of the features.
        series_name : str, optional
            Name of the series being checked, if applicable.

        Returns
        -------
        None

        """
        for feature in not_compliant_features:
            rule = feature_values_range[feature]
            if isinstance(rule, tuple):  # numeric
                msg = (
                    f"'{feature}' has one or more values outside the range seen during training "
                    f"[{rule[0]:.5f}, {rule[1]:.5f}]. "
                    f"This may affect the accuracy of the predictions."
                )
            else:  # categorical
                msg = (
                    f"'{feature}' has values not seen during training. Seen values: "
                    f"{rule}. This may affect the accuracy of the predictions."
                )

            if series_name:
                msg = f"'{series_name}': " + msg
            
            warnings.warn(msg, FeatureOutOfRangeWarning)

    def fit(
            self,
            series: pd.DataFrame | pd.Series | dict,
            exog: pd.DataFrame | pd.Series | dict | None = None
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

        self.series_values_range_ = {}
        self.exog_values_range_ = {}
        self.series_names_in_ = []
        self.exog_names_in_ = []

        if not isinstance(series, (pd.DataFrame, pd.Series, dict)):
            raise TypeError("Input must be a pandas DataFrame, Series or dict.")
        
        if not isinstance(exog, (pd.DataFrame, pd.Series, dict, type(None))):
            raise TypeError("Exogenous variables must be a pandas DataFrame, Series or dict.")

        self.series_dtypes_in_ = type(series)
        if isinstance(series, pd.Series):
            if not series.name:
                raise ValueError("Series must have a name when a pandas Series is provided.")
            series = {series.name: series}
        if isinstance(series, pd.DataFrame):
            if isinstance(series.index, pd.MultiIndex):
                first_col = series.columns[0]
                if len(series.columns) != 1:
                    warnings.warn(
                        f"`series` DataFrame has multiple columns. Only the values of "
                        f"first column, '{first_col}', will be used as series values. "
                        f"All other columns will be ignored.",
                        IgnoredArgumentWarning
                    )
                series = {
                    series_id: series.loc[series_id][first_col].rename(series_id)
                    for series_id in series.index.levels[0]
                }
            else:
                series = {col: series[col] for col in series.columns}

        if isinstance(series, dict):
            for key, value in series.items():
                if not isinstance(value, (pd.Series, pd.DataFrame)):
                    raise TypeError("All values in `series` must be DataFrame or Series.")
                self.series_values_range_[key] = self._get_features_range(X=value)
                self.series_names_in_.append(key)

        if exog is not None:
            self.exog_dtypes_in_ = type(exog)
            if isinstance(exog, pd.Series):
                if not exog.name:
                    raise ValueError("Exog must have a name when a pandas Series is provided.")
                exog = exog.to_frame()
            
            if isinstance(exog, pd.DataFrame):
                if isinstance(exog.index, pd.MultiIndex):
                    exog = {series_id: exog.loc[series_id] for series_id in exog.index.levels[0]}
                else:
                    exog = {col: exog[col] for col in exog.columns}

            
            if isinstance(exog, dict):
                for key, value in exog.items():
                    if not isinstance(value, (pd.Series, pd.DataFrame)):
                        raise TypeError("All values in `exog` must be DataFrame or Series.")
                    self.exog_values_range_[key] = self._get_features_range(X=value)
                if self.exog_dtypes_in_ is dict:
                    self.exog_names_in_ = list(
                        dict.fromkeys(k for v in self.exog_values_range_.values() for k in v)
                    )
                else:
                    self.exog_names_in_ = list(self.exog_values_range_.keys())

        self.is_fitted_ = True

        return

    def predict(
        self,
        last_window: pd.Series | pd.DataFrame | dict | None = None,
        exog: pd.Series | pd.DataFrame | dict | None = None
    ) -> None:
        """
        Check if there is any value outside the training range for last_window and exog.

        Parameters
        ----------
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.

        """

        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet.")

        if not isinstance(last_window, (pd.DataFrame, pd.Series, dict, type(None))):
            raise TypeError("last_window must be a pandas DataFrame, Series, dict or None.")
        
        if not isinstance(exog, (pd.DataFrame, pd.Series, dict, type(None))):
            raise TypeError("Exogenous variables must be a pandas DataFrame, Series, dict or None.")
        
        # if not isinstance(last_window, self.series_dtypes_in_):
        #     raise TypeError(
        #         f"last_window must be of type {self.series_dtypes_in_}. "
        #         f"Got {type(last_window)} instead."
        #     )
        # if not isinstance(exog, self.exog_dtypes_in_):
        #     raise TypeError(
        #         f"exog must be of type {self.exog_dtypes_in_}. "
        #         f"Got {type(exog)} instead."
        #     )

        if last_window is not None:

            if isinstance(last_window, pd.Series):
                if not last_window.name:
                    raise ValueError("last_window Series must have a name.")
                last_window = {last_window.name: last_window}

            elif isinstance(last_window, pd.DataFrame):

                if isinstance(last_window.index, pd.MultiIndex):
                    last_window = {series_id: last_window.loc[series_id] for series_id in last_window.index.levels[0]}
                else:
                    last_window = {col: last_window[col] for col in last_window.columns}

            for key, value in last_window.items():
                
                if not isinstance(value, (pd.Series, pd.DataFrame)):
                    raise TypeError("All values in `last_window` must be DataFrame or Series.")
                
                features_ranges=self.series_values_range_[key]
                if not isinstance(features_ranges, dict):
                    features_ranges = {key: features_ranges}
                
                not_compliant_features = self.check_features_range_(
                    features_ranges=features_ranges,
                    X=value
                )
                self._display_warnings(
                    not_compliant_features=not_compliant_features,
                    feature_values_range=features_ranges,
                    series_name=key
                )

           
        if exog is not None:
            if isinstance(exog, pd.DataFrame) and isinstance(exog.index, pd.MultiIndex):
                exog = {series_id: exog.loc[series_id] for series_id in exog.index.levels[0]}
            self.check_features_range_(self.exog_values_range_, exog, input_name="exog")

        return
