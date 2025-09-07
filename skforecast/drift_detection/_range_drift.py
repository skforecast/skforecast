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
    is_fitted_ : bool
        Whether the detector has been fitted to the training data.
    """

    def __init__(self):
        self.series_names_in_: list[str] = []
        self.series_values_range_: dict | None = None
        self.exog_names_in_: list[str] = []
        self.exog_values_range_: dict | None = None
        self.is_fitted_: bool = False

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
        )

        return info

    @classmethod
    def get_features_range_(cls, X: pd.DataFrame | pd.Series | dict) -> dict:
        """
        Get a summary of the features in the DataFrame or Series. For numeric
        features, it returns the min and max values. For categorical features,
        it returns the unique values.

        Arguments
        ---------
        X : pd.DataFrame, pd.Series, dict
            Input data to summarize. If a dict is provided it should map keys to
            DataFrame or Series objects and the function will return a dict of
            feature summaries for each key.

        Returns
        -------
        features_ranges: dict
            Summary of the features in the input data. If input is a dict, returns
            a dict of dicts mapping the same keys to their feature summaries.

        """

        if not isinstance(X, (pd.DataFrame, pd.Series, dict)):
            raise TypeError("Input must be a pandas DataFrame, Series or dict.")
        
        if isinstance(X, pd.Series):
            X = X.to_frame(name=X.name if X.name is not None else 'y')

        if isinstance(X, dict):
            features_ranges = {}
            for key, value in X.items():
                if not isinstance(value, (pd.DataFrame, pd.Series)):
                    raise TypeError("All dictionary values must be DataFrame or Series.")
                if isinstance(value, pd.Series):
                    value = value.to_frame(name=key)
                
                num_cols = [col for col, dt in value.dtypes.items() if pd.api.types.is_numeric_dtype(dt)]
                cat_cols = [col for col in value.columns if col not in num_cols]

                ranges = {}
                ranges.update({col: (value[col].min(), value[col].max()) for col in num_cols})
                ranges.update({col: set(value[col].dropna().unique()) for col in cat_cols})

                features_ranges[key] = ranges

        if isinstance(X, pd.DataFrame):
            num_cols = [col for col, dt in X.dtypes.items() if pd.api.types.is_numeric_dtype(dt)]
            cat_cols = [col for col in X.columns if col not in num_cols]

            features_ranges = {}
            features_ranges.update({col: (X[col].min(), X[col].max()) for col in num_cols})
            features_ranges.update({col: set(X[col].dropna().unique()) for col in cat_cols})

        return features_ranges

    @classmethod
    def check_features_range_(
        cls,
        features_ranges: dict, 
        X: pd.DataFrame | pd.Series | dict,
        input_name: str = None
    ) -> None:
        """
        Check if there is any value outside the training range. For numeric features,
        it checks if the values are within the min and max range. For categorical features,
        it checks if the values are among the seen categories.

        Parameters
        ----------
        features_ranges : dict
            Output from get_features_range_()
        X : pd.DataFrame, pd.Series, or dict
            New data to validate
        input_name : str, optional
            Name to prepend to warning messages if checking a dict of inputs.
        """

        to_check = []
        is_exog = input_name == 'exog'

        if isinstance(X, dict):
            for key, value in X.items():
                if isinstance(value, pd.Series):
                    value = value.to_frame(name=value.name if value.name else 'y')
                elif not isinstance(value, pd.DataFrame):
                    raise TypeError("All dictionary values must be DataFrame or Series.")
                to_check.append((key if is_exog else key, value))
        elif isinstance(X, pd.Series):
            to_check.append((input_name or 'X', X.to_frame(name=X.name if X.name else 'y')))
        elif isinstance(X, pd.DataFrame):
            to_check.append((input_name or 'X', X))
        else:
            raise TypeError("Input must be a pandas DataFrame, Series, or dict.")

        for name, df in to_check:
            if is_exog and isinstance(X, dict):
                ranges = features_ranges[name]
            else:
                ranges = features_ranges

            for col in set(df.columns).intersection(ranges.keys()):
                rule = ranges[col]

                if isinstance(rule, tuple):  # numeric
                    if df[col].min() < rule[0] or df[col].max() > rule[1]:
                        if is_exog and isinstance(X, dict):
                            label = f"exog '{col}'"
                        else:
                            label = f"'{col}'"

                        msg = (
                            f"{label} has one or more values outside the range seen during training "
                            f"[{rule[0]:.5f}, {rule[1]:.5f}]. "
                            f"This may affect the accuracy of the predictions."
                        )
                        if name:
                            msg = f"'{name}': " + msg
                        warnings.warn(msg, FeatureOutOfRangeWarning)
                else:  # categorical
                    unseen = set(df[col].unique()) - rule
                    if unseen:
                        msg = (
                            f"'{col}' has values not seen during training. Seen values: "
                            f"{rule}. This may affect the accuracy of the predictions."
                        )
                        if name:
                            msg = f"'{name}': " + msg
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

        if not isinstance(series, (pd.DataFrame, pd.Series, dict)):
            raise TypeError("Input must be a pandas DataFrame, Series or dict.")
        
        if not isinstance(exog, (pd.DataFrame, pd.Series, dict, type(None))):
            raise TypeError("Exogenous variables must be a pandas DataFrame, Series or dict.")

        if isinstance(series, pd.DataFrame) and isinstance(series.index, pd.MultiIndex):
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

        self.series_values_range_ = self.get_features_range_(X=series)
        if isinstance(series, dict):
            self.series_values_range_ = {
                k: v[k] for k, v in self.series_values_range_.items()
            }
        self.series_names_in_ = list(self.series_values_range_.keys())

        if exog is not None:
            if isinstance(exog, pd.DataFrame) and isinstance(exog.index, pd.MultiIndex):
                exog = {series_id: exog.loc[series_id] for series_id in exog.index.levels[0]}
            
            self.exog_values_range_ = self.get_features_range_(X=exog)
            
            if any(isinstance(v, dict) for v in self.exog_values_range_.values()):
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

        if last_window is not None:
            self.check_features_range_(self.series_values_range_, last_window, input_name="last_window")

        if exog is not None:
            if isinstance(exog, pd.DataFrame) and isinstance(exog.index, pd.MultiIndex):
                exog = {series_id: exog.loc[series_id] for series_id in exog.index.levels[0]}
            self.check_features_range_(self.exog_values_range_, exog, input_name="exog")

        return
