################################################################################
#                             RangeDriftDetector                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import pandas as pd
import numpy as np
import warnings
from ..exceptions import FeatureOutOfRangeWarning


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
        self.series_values_range_: dict | None = None
        self.exog_values_range_: dict | None = None
        self.is_fitted_: bool = False

    @classmethod
    def get_features_range(cls, X: pd.DataFrame | pd.Series | dict) -> dict:
        """
        Get a summary of the features in the DataFrame or Series. For numeric features,
        it returns the min and max values. For categorical features, it returns the
        unique values.

        Arguments
        ---------
        X : pd.DataFrame or pd.Series or dict
            Input data to summarize. If a dict is provided it should map keys to
            DataFrame or Series objects and the function will return a dict of
            feature summaries for each key.

        Returns
        -------
        dict
            Summary of the features in the input data. If input is a dict, returns
            a dict of dicts mapping the same keys to their feature summaries.

        """

        if not isinstance(X, (pd.DataFrame, pd.Series, dict)):
            raise TypeError("Input must be a pandas DataFrame, Series or dict.")

        if isinstance(X, dict):
            return {key: cls.get_features_range(series) for key, series in X.items()}

        if isinstance(X, pd.Series):
            X = X.to_frame(name=X.name if X.name is not None else 'y')

        num_cols = [col for col, dt in X.dtypes.items() if pd.api.types.is_numeric_dtype(dt)]
        cat_cols = [col for col in X.columns if col not in num_cols]

        features_ranges = {col: (X[col].min(), X[col].max()) for col in num_cols}
        features_ranges.update({col: set(X[col].dropna().unique()) for col in cat_cols})

        return features_ranges

    @classmethod
    def check_features_range(
        cls,
        features_ranges: dict, X: pd.DataFrame | pd.Series,
        input_name: str=None
    ) -> bool:
        """
        Check if there is any value outside the training range. For numeric features,
        it checks if the values are within the min and max range. For categorical features,
        it checks if the values are among the seen categories.

        Parameters
        ----------
        features_ranges : dict
            Output from get_feature_summary()
        X : pd.DataFrame or pd.Series
            New data to validate
        """

        if isinstance(X, dict):
            for key, v in X.items():
                cls.check_features_range(features_ranges[key], v, series_name=key)
                return

        if isinstance(X, pd.Series):
            X = X.to_frame(name=X.name if X.name is not None else 'y')

        for col in set(X.columns).intersection(features_ranges.keys()):
            rule = features_ranges[col]
            if isinstance(rule, tuple):  # numeric
                if X[col].min() < rule[0] or X[col].max() > rule[1]:
                    msg = (
                        f"'{col}' has one or more values outside the range seen during training "
                        f"[{rule[0]:.5f}, {rule[1]:.5f}]. "
                        f"This may affect the accuracy of the predictions."
                    )
                    if input_name:
                        msg = f"'{input_name}': " + msg
                    warnings.warn(msg, FeatureOutOfRangeWarning)
            else:
                unseen = set(X[col].unique()) - rule
                if unseen:
                    msg = (
                        f"'{col}' has values not seen during training. Seen values: "
                        f"{rule}. This may affect the accuracy of the predictions."
                    )
                    if input_name:
                        msg = f"'{input_name}': " + msg
                    warnings.warn(msg, FeatureOutOfRangeWarning)
        return



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

        self.series_values_range_ = self.get_features_range(X=series)
        if exog is not None:
            self.exog_values_range_ = self.get_features_range(X=exog)

        self.is_fitted_ = True

        return

    def predict(
        self,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
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
        
        if not isinstance(last_window, (pd.DataFrame, pd.Series, type(None))):
            raise TypeError("last_window must be a pandas DataFrame, Series or None.")
        
        if not isinstance(exog, (pd.DataFrame, pd.Series, type(None))):
            raise TypeError("exog must be a pandas DataFrame, Series or None.")

        self.check_features_range(self.series_values_range_, last_window, input_name="last_window")
        self.check_features_range(self.exog_values_range_, exog, input_name="exog")

        return
