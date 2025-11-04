################################################################################
#                           ForecasterRecursiveClassifier                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Callable, Any
import warnings
import sys
import numpy as np
import pandas as pd
from copy import copy
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import clone

import skforecast
from ..base import ForecasterBase
from ..exceptions import DataTransformationWarning
from ..utils import (
    initialize_lags,
    initialize_window_features,
    initialize_weights,    
    check_select_fit_kwargs,
    check_y,
    check_exog,
    get_exog_dtypes,
    check_exog_dtypes,
    check_predict_input,
    check_extract_values_and_index,
    input_to_frame,
    date_to_index_position,
    expand_index,
    transform_dataframe,
    get_style_repr_html,
    set_cpu_gpu_device
)


# TODO: Calibrate?
# TODO: TunedThresholdClassifierCV? It is only for binary classification
class ForecasterRecursiveClassifier(ForecasterBase):
    """
    This class turns any classifier compatible with the scikit-learn API into a
    recursive autoregressive (multi-step) forecaster.
    
    Parameters
    ----------
    regressor : classifier or pipeline compatible with the scikit-learn API
        An instance of a classifier or pipeline compatible with the scikit-learn API.
    lags : int, list, numpy ndarray, range, default None
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
    
        - `int`: include lags from 1 to `lags` (included).
        - `list`, `1d numpy ndarray` or `range`: include only lags present in 
        `lags`, all elements must be int.
        - `None`: no lags are included as predictors. 
    window_features : object, list, default None
        Instance or list of instances used to create window features. Window features
        are created from the original time series and are included as predictors.
    features_encoding : str, default 'auto'
        Encoding method for features derived from the time series (lags and 
        window features that return class values):
        
        - 'auto': Use categorical dtype if classifier supports native categorical
        features (LightGBM, CatBoost, XGBoost), otherwise numeric encoding.
        - 'categorical': Force categorical dtype (requires compatible classifier).
        - 'ordinal': Use ordinal encoding (0, 1, 2, ...). The classifier will 
        treat class codes as numeric values, assuming an ordinal relationship 
        between classes (e.g., 'low' < 'medium' < 'high').
        
        Note: This only affects features derived from the target series (y) not 
        exogenous variables.
    transformer_exog : object transformer (preprocessor), default None
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable, default None
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method. The resulting `sample_weight` cannot have negative values.
    fit_kwargs : dict, default None
        Additional arguments to be passed to the `fit` method of the classifier.
    forecaster_id : str, int, default None
        Name used as an identifier of the forecaster.
    
    Attributes
    ----------
    regressor : classifier or pipeline compatible with the scikit-learn API
        An instance of a classifier or pipeline compatible with the scikit-learn API.
    lags : numpy ndarray
        Lags used as predictors.
    lags_names : list
        Names of the lags used as predictors.
    max_lag : int
        Maximum lag included in `lags`.
    window_features : list
        Class or list of classes used to create window features.
    window_features_names : list
        Names of the window features to be included in the `X_train` matrix.
    window_features_class_names : list
        Names of the classes used to create the window features.
    max_size_window_features : int
        Maximum window size required by the window features.
    window_size : int
        The window size needed to create the predictors. It is calculated as the 
        maximum value between `max_lag` and `max_size_window_features`.
    features_encoding : str
        Encoding method for features derived from the time series (lags and 
        window features that return class values).
    use_native_categoricals : bool
        Indicates whether the classifier supports native categorical features.
    classes_ : list
        List of class labels seen during training.
    class_codes_ : list
        List of class codes assigned by the `OrdinalEncoder` during training.
    n_classes_ : int
        Number of classes seen during training.
    encoder : OrdinalEncoder
        Instance of `OrdinalEncoder` used to encode target variable class labels.
    encoding_mapping_ : dict
        Mapping of original class labels to encoded values.
    code_to_class_mapping_ : dict
        Mapping of encoded values to original class labels.
    transformer_exog : object transformer (preprocessor)
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `classifier` does not have the argument `sample_weight` in its `fit`
        method. The resulting `sample_weight` cannot have negative values.
    source_code_weight_func : str
        Source code of the custom function used to create weights.
    last_window_ : pandas DataFrame
        This window represents the most recent data observed by the predictor
        during its training phase. It contains the values needed to predict the
        next step immediately after the training data. These values are stored
        in the original scale of the time series before undergoing any transformation.
    index_type_ : type
        Type of index of the input used in training.
    index_freq_ : str
        Frequency of Index of the input used in training.
    training_range_ : pandas Index
        First and last values of index of the data used during training.
    series_name_in_ : str
        Names of the series provided by the user during training.
    exog_in_ : bool
        If the forecaster has been trained using exogenous variable/s.
    exog_names_in_ : list
        Names of the exogenous variables used during training.
    exog_type_in_ : type
        Type of exogenous data (pandas Series or DataFrame) used in training.
    exog_dtypes_in_ : dict
        Type of each exogenous variable/s used in training before the transformation
        applied by `transformer_exog`. If `transformer_exog` is not used, it
        is equal to `exog_dtypes_out_`.
    exog_dtypes_out_ : dict
        Type of each exogenous variable/s used in training after the transformation 
        applied by `transformer_exog`. If `transformer_exog` is not used, it 
        is equal to `exog_dtypes_in_`.
    X_train_window_features_names_out_ : list
        Names of the window features included in the matrix `X_train` created
        internally for training.
    X_train_exog_names_out_ : list
        Names of the exogenous variables included in the matrix `X_train` created
        internally for training. It can be different from `exog_names_in_` if
        some exogenous variables are transformed during the training process.
    X_train_features_names_out_ : list
        Names of columns of the matrix created internally for training.
    fit_kwargs : dict
        Additional arguments to be passed to the `fit` method of the classifier.
    creation_date : str
        Date of creation.
    is_fitted : bool
        Tag to identify if the classifier has been fitted (trained).
    fit_date : str
        Date of last fit.
    skforecast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    forecaster_id : str, int
        Name used as an identifier of the forecaster.
    __skforecast_tags__ : dict
        Tags associated with the forecaster.
    _probabilistic_mode: str, bool
        Private attribute used to indicate whether the forecaster should perform 
        some calculations during backtesting.

    Notes
    -----
    Categorical features are transformed using an `OrdinalEncoder` (self.encoder).
    The encoder's learned mappings (self.encoding_mapping_) are stored so that 
    later, when creating lag (autoregressive) features, the same category-to-integer 
    relationships can be applied consistently.

    The goal is to ensure that the lag features — which are recreated as 
    categorical variables — use the exact same integer codes as the original encoding.
    In other words, the numerical values in the lagged features should 
    exactly match the integer codes that the `OrdinalEncoder` assigned.
    Formally, this means the following should hold true:

    `(X_train['lag_1'].cat.codes == X_train['lag_1']).all()`

    This consistency is guaranteed because:

    - `OrdinalEncoder` assigns integer codes starting from 0, in the alphabetical 
    order of category labels.

    - When autoregressive (lag) features are created later, they are converted 
    to pandas Categorical types using the same category ordering 
    (`categories = forecaster.class_codes_`).

    As a result, the categorical codes used in lag features remain aligned
    with the original encoding from the `OrdinalEncoder`.

    During prediction, we can work directly with NumPy arrays because the 
    `OrdinalEncoder` transforms new observations into the same integer codes 
    used by pandas Categorical during training. This eliminates the need to 
    convert data to pandas categorical types at inference time.
    
    """

    def __init__(
        self,
        regressor: object,
        lags: int | list[int] | np.ndarray[int] | range[int] | None = None,
        window_features: object | list[object] | None = None,
        features_encoding: str = 'auto',
        transformer_exog: object | None = None,
        weight_func: Callable | None = None,
        fit_kwargs: dict[str, object] | None = None,
        forecaster_id: str | int | None = None
    ) -> None:
        
        self.regressor                          = copy(regressor)
        self.transformer_exog                   = transformer_exog
        self.weight_func                        = weight_func
        self.source_code_weight_func            = None
        self.last_window_                       = None
        self.index_type_                        = None
        self.index_freq_                        = None
        self.training_range_                    = None
        self.series_name_in_                    = None
        self.exog_in_                           = False
        self.exog_names_in_                     = None
        self.exog_type_in_                      = None
        self.exog_dtypes_in_                    = None
        self.exog_dtypes_out_                   = None
        self.X_train_window_features_names_out_ = None
        self.X_train_exog_names_out_            = None
        self.X_train_features_names_out_        = None
        self.creation_date                      = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.is_fitted                          = False
        self.fit_date                           = None
        self.skforecast_version                 = skforecast.__version__
        self.python_version                     = sys.version.split(" ")[0]
        self.forecaster_id                      = forecaster_id
        self._probabilistic_mode                = "binned"  # TODO: Check

        self.features_encoding                  = features_encoding
        self.use_native_categoricals            = False
        self.classes_                           = None
        self.class_codes_                       = None
        self.n_classes_                         = None
        self.encoding_mapping_                  = None
        self.code_to_class_mapping_             = None

        valid_encodings = ['auto', 'categorical', 'ordinal']
        if features_encoding not in valid_encodings:
            raise ValueError(
                f"`features_encoding` must be one of {valid_encodings}. "
                f"Got '{features_encoding}'."
            )
        
        supports_categorical = self._check_categorical_support(regressor)
        if features_encoding == 'categorical':
            if supports_categorical:
                self.use_native_categoricals = True
            else:
                raise ValueError(
                    f"`features_encoding='categorical'` requires a classifier that "
                    f"supports native categorical features (LightGBM, CatBoost, XGBoost). "
                    f"Got {type(regressor).__name__}. Use 'auto' or 'ordinal' instead."
                )
        elif features_encoding == 'auto':
            if supports_categorical:
                self.use_native_categoricals = True

        self.encoder = OrdinalEncoder(
                           categories = 'auto',
                           dtype      = float if features_encoding == 'ordinal' else int
                       )

        self.lags, self.lags_names, self.max_lag = initialize_lags(type(self).__name__, lags)
        self.window_features, self.window_features_names, self.max_size_window_features = (
            initialize_window_features(window_features)
        )
        if self.window_features is None and self.lags is None:
            raise ValueError(
                "At least one of the arguments `lags` or `window_features` "
                "must be different from None. This is required to create the "
                "predictors used in training the forecaster."
            )
        
        self.window_size = max(
            [ws for ws in [self.max_lag, self.max_size_window_features] 
             if ws is not None]
        )
        self.window_features_class_names = None
        if window_features is not None:
            self.window_features_class_names = [
                type(wf).__name__ for wf in self.window_features
            ]

        self.weight_func, self.source_code_weight_func, _ = initialize_weights(
            forecaster_name = type(self).__name__, 
            regressor       = regressor, 
            weight_func     = weight_func, 
            series_weights  = None
        )

        self.fit_kwargs = check_select_fit_kwargs(
                              regressor  = regressor,
                              fit_kwargs = fit_kwargs
                          )
        
        self.__skforecast_tags__ = {
            "library": "skforecast",
            "estimator_type": "forecaster",
            "estimator_name": "ForecasterRecursiveClassifier",
            "estimator_task": "classification",
            "forecasting_scope": "single-series",  # single-series | global
            "forecasting_strategy": "recursive",   # recursive | direct | deep_learning
            "index_types_supported": ["pandas.RangeIndex", "pandas.DatetimeIndex"],
            "requires_index_frequency": True,

            "allowed_input_types_series": ["pandas.Series"],
            "supports_exog": True,
            "allowed_input_types_exog": ["pandas.Series", "pandas.DataFrame"],
            "handles_missing_values_series": False, 
            "handles_missing_values_exog": True, 

            "supports_lags": True,
            "supports_window_features": True,
            "supports_transformer_series": False,
            "supports_transformer_exog": True,
            "supports_weight_func": True,
            "supports_differentiation": False,

            "prediction_types": ["point", "probabilities"],
            "supports_probabilistic": True,
            "probabilistic_methods": ["class-probabilities"],
            "handles_binned_residuals": False
        }


    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a ForecasterRecursive object is printed.
        """

        (
            params,
            _,
            _,
            exog_names_in_,
            _,
        ) = self._preprocess_repr(
                regressor      = self.regressor,
                exog_names_in_ = self.exog_names_in_
            )
        
        params = self._format_text_repr(params)
        exog_names_in_ = self._format_text_repr(exog_names_in_)

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Classifier: {type(self.regressor).__name__} \n"
            f"Lags: {self.lags} \n"
            f"Window features: {self.window_features_names} \n"
            f"Window size: {self.window_size} \n"
            f"Series name: {self.series_name_in_} \n"
            f"Classes: {self.classes_} \n"
            f"Number of classes: {self.n_classes_} \n"
            f"Exogenous included: {self.exog_in_} \n"
            f"Exogenous names: {exog_names_in_} \n"
            f"Feature encoding: {self.features_encoding} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Weight function included: {True if self.weight_func is not None else False} \n"
            f"Training range: {self.training_range_.to_list() if self.is_fitted else None} \n"
            f"Training index type: {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else None} \n"
            f"Training index frequency: {self.index_freq_ if self.is_fitted else None} \n"
            f"Classifier parameters: {params} \n"
            f"fit_kwargs: {self.fit_kwargs} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Skforecast version: {self.skforecast_version} \n"
            f"Python version: {self.python_version} \n"
            f"Forecaster id: {self.forecaster_id} \n"
        )

        return info

    # TODO: update user guide
    def _repr_html_(self):
        """
        HTML representation of the object.
        The "General Information" section is expanded by default.
        """

        (
            params,
            _,
            _,
            exog_names_in_,
            _,
        ) = self._preprocess_repr(
                regressor      = self.regressor,
                exog_names_in_ = self.exog_names_in_
            )

        style, unique_id = get_style_repr_html(self.is_fitted)
        
        content = f"""
        <div class="container-{unique_id}">
            <h2>{type(self).__name__}</h2>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Classifier:</strong> {type(self.regressor).__name__}</li>
                    <li><strong>Lags:</strong> {self.lags}</li>
                    <li><strong>Window features:</strong> {self.window_features_names}</li>
                    <li><strong>Window size:</strong> {self.window_size}</li>
                    <li><strong>Series name:</strong> {self.series_name_in_}</li>
                    <li><strong>Exogenous included:</strong> {self.exog_in_}</li>
                    <li><strong>Weight function included:</strong> {self.weight_func is not None}</li>
                    <li><strong>Creation date:</strong> {self.creation_date}</li>
                    <li><strong>Last fit date:</strong> {self.fit_date}</li>
                    <li><strong>Skforecast version:</strong> {self.skforecast_version}</li>
                    <li><strong>Python version:</strong> {self.python_version}</li>
                    <li><strong>Forecaster id:</strong> {self.forecaster_id}</li>
                </ul>
            </details>
            <details>
                <summary>Classification Information</summary>
                <ul>
                    <li><strong>Classes:</strong> {self.classes_}</li>
                    <li><strong>Class encoding:</strong> {self.encoding_mapping_}</li>
                </ul>
            </details>
            <details>
                <summary>Exogenous Variables</summary>
                <ul>
                    <li><strong>Exogenous names:</strong> {exog_names_in_}</li>
                    <li><strong>Transformer for exog:</strong> {self.transformer_exog}</li>
                </ul>
            </details>
            <details>
                <summary>Training Information</summary>
                <ul>
                    <li><strong>Training range:</strong> {self.training_range_.to_list() if self.is_fitted else 'Not fitted'}</li>
                    <li><strong>Training index type:</strong> {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else 'Not fitted'}</li>
                    <li><strong>Training index frequency:</strong> {self.index_freq_ if self.is_fitted else 'Not fitted'}</li>
                </ul>
            </details>
            <details>
                <summary>Classifier Parameters</summary>
                <ul>
                    {params}
                </ul>
            </details>
            <details>
                <summary>Fit Kwargs</summary>
                <ul>
                    {self.fit_kwargs}
                </ul>
            </details>
            <p>
                <a href="https://skforecast.org/{skforecast.__version__}/api/forecasterrecursiveclassifier.html">&#128712 <strong>API Reference</strong></a>
                &nbsp;&nbsp;
                <a href="https://skforecast.org/{skforecast.__version__}/user_guides/forecasting-classification.html">&#128462 <strong>User Guide</strong></a>
            </p>
        </div>
        """

        return style + content
    
    def _check_categorical_support(
        self, 
        regressor: object
    ) -> bool:
        """
        Check if classifier supports native categorical features.
        Checks by class name to avoid importing optional dependencies.
        """

        if isinstance(regressor, Pipeline):
            estimator = regressor[-1]
        else:
            estimator = regressor
        
        class_name = type(estimator).__name__
        module_name = type(estimator).__module__
        
        supported_models = {
            'LGBMClassifier': 'lightgbm',
            'CatBoostClassifier': 'catboost',
            'XGBClassifier': 'xgboost',
            'HistGradientBoostingClassifier': 'sklearn.ensemble._hist_gradient_boosting'
        }
        
        if class_name in supported_models:
            expected_module = supported_models[class_name]
            # NOTE: Verify if the estimator is from the expected module
            # (in case someone creates a class with the same name)
            if expected_module in module_name:
                return True
        
        return False

    def _create_lags(
        self,
        y: np.ndarray,
        X_as_pandas: bool = False,
        train_index: pd.Index | None = None,
        class_codes: list[int | float] | None = None
    ) -> tuple[np.ndarray | pd.DataFrame | None, np.ndarray]:
        """
        Create the lagged values and their target variable from a time series.
        
        Note that the returned matrix `X_data` contains the lag 1 in the first 
        column, the lag 2 in the in the second column and so on.
        
        Parameters
        ----------
        y : numpy ndarray
            Training time series values.
        X_as_pandas : bool, default False
            If `True`, the returned matrix `X_data` is a pandas DataFrame.
        train_index : pandas Index, default None
            Index of the training data. It is used to create the pandas DataFrame
            `X_data` when `X_as_pandas` is `True`.
        class_codes : list, default None
            List of category codes to be used when converting lagged values to
            pandas Categorical. Only used when `self.use_native_categoricals` is 
            `True`.

        Returns
        -------
        X_data : numpy ndarray, pandas DataFrame, None
            Lagged values (predictors).
        y_data : numpy ndarray
            Values of the time series related to each row of `X_data`.
        
        Notes
        -----
        Returned matrices are views into the original `y` so care must be taken
        when modifying them.

        """
        
        X_data = None
        if self.lags is not None:
            y_strided = np.lib.stride_tricks.sliding_window_view(y, self.window_size)[:-1]
            X_data = y_strided[:, self.window_size - self.lags]

            if X_as_pandas:
                X_data = pd.DataFrame(
                             data    = X_data,
                             columns = self.lags_names,
                             index   = train_index
                         )
                if self.use_native_categoricals:
                    for col in X_data.columns:
                        X_data[col] = pd.Categorical(
                                          values     = X_data[col],
                                          categories = class_codes,
                                          ordered    = False
                                      )

        y_data = y[self.window_size:]

        return X_data, y_data

    def _create_window_features(
        self, 
        y: pd.Series,
        train_index: pd.Index,
        X_as_pandas: bool = False,
    ) -> tuple[list[np.ndarray | pd.DataFrame], list[str]]:
        """
        Create window features from a time series.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        train_index : pandas Index
            Index of the training data. It is used to create the pandas DataFrame
            `X_train_window_features` when `X_as_pandas` is `True`.
        X_as_pandas : bool, default False
            If `True`, the returned matrix `X_train_window_features` is a 
            pandas DataFrame.

        Returns
        -------
        X_train_window_features : list
            List of numpy ndarrays or pandas DataFrames with the window features.
        X_train_window_features_names_out_ : list
            Names of the window features.
        
        """

        len_train_index = len(train_index)
        X_train_window_features = []
        X_train_window_features_names_out_ = []
        for wf in self.window_features:
            X_train_wf = wf.transform_batch(y)
            if not isinstance(X_train_wf, pd.DataFrame):
                raise TypeError(
                    f"The method `transform_batch` of {type(wf).__name__} "
                    f"must return a pandas DataFrame."
                )
            X_train_wf = X_train_wf.iloc[-len_train_index:]
            if not len(X_train_wf) == len_train_index:
                raise ValueError(
                    f"The method `transform_batch` of {type(wf).__name__} "
                    f"must return a DataFrame with the same number of rows as "
                    f"the input time series - `window_size`: {len_train_index}."
                )
            if not (X_train_wf.index == train_index).all():
                raise ValueError(
                    f"The method `transform_batch` of {type(wf).__name__} "
                    f"must return a DataFrame with the same index as "
                    f"the input time series - `window_size`."
                )
            
            X_train_window_features_names_out_.extend(X_train_wf.columns)
            if not X_as_pandas:
                X_train_wf = X_train_wf.to_numpy()     
            X_train_window_features.append(X_train_wf)

        return X_train_window_features, X_train_window_features_names_out_


    def _create_train_X_y(
        self,
        y: pd.Series,
        exog: pd.Series | pd.DataFrame | None = None,
        store_last_window: bool | list[str] = True
    ) -> tuple[
        pd.DataFrame, 
        pd.Series, 
        list[str], 
        list[str], 
        list[str], 
        list[str], 
        dict[str, type],
        dict[str, type]
    ]:
        """
        Create training matrices from univariate time series and exogenous
        variables.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned.
        store_last_window : bool, default True
            Whether or not to store the last window (`last_window_`) of training data.

        Returns
        -------
        X_train : pandas DataFrame
            Training values (predictors).
        y_train : pandas Series
            Values of the time series related to each row of `X_train`.
        exog_names_in_ : list
            Names of the exogenous variables used during training.
        X_train_window_features_names_out_ : list
            Names of the window features included in the matrix `X_train` created
            internally for training.
        X_train_exog_names_out_ : list
            Names of the exogenous variables included in the matrix `X_train` created
            internally for training. It can be different from `exog_names_in_` if
            some exogenous variables are transformed during the training process.
        X_train_features_names_out_ : list
            Names of the columns of the matrix created internally for training.
        exog_dtypes_in_ : dict
            Type of each exogenous variable/s used in training before the transformation
            applied by `transformer_exog`. If `transformer_exog` is not used, it
            is equal to `exog_dtypes_out_`.
        exog_dtypes_out_ : dict
            Type of each exogenous variable/s used in training after the transformation
            applied by `transformer_exog`. If `transformer_exog` is not used, it 
            is equal to `exog_dtypes_in_`.

        """

        check_y(y=y)
        y = input_to_frame(data=y, input_name='y')

        if len(y) <= self.window_size:
            raise ValueError(
                f"Length of `y` must be greater than the maximum window size "
                f"needed by the forecaster.\n"
                f"    Length `y`: {len(y)}.\n"
                f"    Max window size: {self.window_size}.\n"
                f"    Lags window size: {self.max_lag}.\n"
                f"    Window features window size: {self.max_size_window_features}."
            )
        
        y_values, y_index = check_extract_values_and_index(data=y, data_label='`y`')

        if np.issubdtype(y_values.dtype, np.floating):
            not_allowed = np.mod(y_values, 1) != 0
            if np.any(not_allowed):
                examples = ", ".join(map(str, np.unique(y_values[not_allowed])[:5]))
                raise ValueError(
                    f"Invalid target for classification: targets must be discrete "
                    f"class labels (strings, integers or floats with decimals "
                    f"equal to 0). Received float dtype '{y_values.dtype}' with "
                    f"decimals (e.g., {examples}). "
                )

        # NOTE: See Notes sections for explanation
        fit_transformer = False if self.is_fitted else True
        if fit_transformer:
            encoding_mapping_ = {}
            y_encoded = self.encoder.fit_transform(y_values.reshape(-1, 1)).ravel()
            for i, cat in enumerate(self.encoder.categories_[0]):
                encoding_mapping_[cat] = i if self.features_encoding != 'ordinal' else float(i)
        else:
            encoding_mapping_ = self.encoding_mapping_
            y_encoded = self.encoder.transform(y_values.reshape(-1, 1)).ravel()

        classes = list(encoding_mapping_.keys())
        class_codes = list(encoding_mapping_.values())
        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError(
                f"The target variable must have at least 2 classes. "
                f"Found {classes} class."
            )
        
        y_encoding_info_ = {
            'classes_': classes,
            'class_codes_': class_codes,
            'n_classes_': n_classes,
            'encoding_mapping_': encoding_mapping_
        }
        train_index = y_index[self.window_size:]

        exog_names_in_ = None
        exog_dtypes_in_ = None
        exog_dtypes_out_ = None
        X_as_pandas = False if not self.use_native_categoricals else True
        if exog is not None:
            check_exog(exog=exog, allow_nan=True)
            exog = input_to_frame(data=exog, input_name='exog')
            _, exog_index = check_extract_values_and_index(
                data=exog, data_label='`exog`', ignore_freq=True, return_values=False
            )

            len_y = len(y_values)
            len_train_index = len(train_index)
            len_exog = len(exog)
            if not len_exog == len_y and not len_exog == len_train_index:
                raise ValueError(
                    f"Length of `exog` must be equal to the length of `y` (if index is "
                    f"fully aligned) or length of `y` - `window_size` (if `exog` "
                    f"starts after the first `window_size` values).\n"
                    f"    `exog`              : ({exog_index[0]} -- {exog_index[-1]})  (n={len_exog})\n"
                    f"    `y`                 : ({y.index[0]} -- {y.index[-1]})  (n={len_y})\n"
                    f"    `y` - `window_size` : ({train_index[0]} -- {train_index[-1]})  (n={len_train_index})"
                )

            exog_names_in_ = exog.columns.to_list()
            exog_dtypes_in_ = get_exog_dtypes(exog=exog)

            exog = transform_dataframe(
                       df                = exog,
                       transformer       = self.transformer_exog,
                       fit               = fit_transformer,
                       inverse_transform = False
                   )
            
            check_exog_dtypes(exog, call_check_exog=True)
            exog_dtypes_out_ = get_exog_dtypes(exog=exog)
            if X_as_pandas is False:
                X_as_pandas = any(
                    not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) 
                    for dtype in set(exog.dtypes)
                )

            if len_exog == len_y:
                if not (exog_index == y_index).all():
                    raise ValueError(
                        "When `exog` has the same length as `y`, the index of "
                        "`exog` must be aligned with the index of `y` "
                        "to ensure the correct alignment of values."
                    )
                # The first `self.window_size` positions have to be removed from 
                # exog since they are not in X_train.
                exog = exog.iloc[self.window_size:, ]
            else:
                if not (exog_index == train_index).all():
                    raise ValueError(
                        "When `exog` doesn't contain the first `window_size` observations, "
                        "the index of `exog` must be aligned with the index of `y` minus "
                        "the first `window_size` observations to ensure the correct "
                        "alignment of values."
                    )
            
        X_train = []
        X_train_features_names_out_ = []

        X_train_lags, y_train = self._create_lags(
                                    y           = y_encoded, 
                                    X_as_pandas = X_as_pandas, 
                                    train_index = train_index,
                                    class_codes = class_codes
                                )
        if X_train_lags is not None:
            X_train.append(X_train_lags)
            X_train_features_names_out_.extend(self.lags_names)
        
        X_train_window_features_names_out_ = None
        if self.window_features is not None:
            y_window_features = pd.Series(y_encoded, index=y_index)
            X_train_window_features, X_train_window_features_names_out_ = (
                self._create_window_features(
                    y=y_window_features, X_as_pandas=X_as_pandas, train_index=train_index
                )
            )

            # FIXME: When 'mode' is used, ideally it should be converted to categorical
            # not done as we can't know its position when 'proportion' is used.

            X_train.extend(X_train_window_features)
            X_train_features_names_out_.extend(X_train_window_features_names_out_)

        X_train_exog_names_out_ = None
        if exog is not None:
            X_train_exog_names_out_ = exog.columns.to_list()  
            if not X_as_pandas:
                exog = exog.to_numpy()     
            X_train_features_names_out_.extend(X_train_exog_names_out_)
            X_train.append(exog)
        
        if len(X_train) == 1:
            X_train = X_train[0]
        else:
            if X_as_pandas:
                X_train = pd.concat(X_train, axis=1)
            else:
                X_train = np.concatenate(X_train, axis=1)
                
        if X_as_pandas:
            X_train.index = train_index
        else:
            X_train = pd.DataFrame(
                          data    = X_train,
                          index   = train_index,
                          columns = X_train_features_names_out_
                      )
        
        y_train = pd.Series(
                      data  = y_train,
                      index = train_index,
                      name  = 'y'
                  )

        last_window_ = None
        if store_last_window:
            last_window_ = pd.DataFrame(
                               data    = y_values[-self.window_size:],
                               index   = y_index[-self.window_size:],
                               columns = y.columns   
                           )

        return (
            X_train,
            y_train,
            y_encoding_info_,
            exog_names_in_,
            X_train_window_features_names_out_,
            X_train_exog_names_out_,
            X_train_features_names_out_,
            exog_dtypes_in_,
            exog_dtypes_out_,
            last_window_
        )
    
    def create_train_X_y(
        self,
        y: pd.Series,
        exog: pd.Series | pd.DataFrame | None = None,
        encoded: bool = True
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Create training matrices from univariate time series and exogenous
        variables.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned.
        encoded : bool, default True
            Whether to return the target and lag features encoded as integers
            (as used during training) or decoded to their original categories.

        Returns
        -------
        X_train : pandas DataFrame
            Training values (predictors).
        y_train : pandas Series
            Values of the time series related to each row of `X_data`.

        Notes
        -----
        Categorical features are transformed using an `OrdinalEncoder` (self.encoder).
        The encoder's learned mappings (self.encoding_mapping_) are stored so that 
        later, when creating lag (autoregressive) features, the same category-to-integer 
        relationships can be applied consistently.

        The goal is to ensure that the lag features — which are recreated as 
        categorical variables — use the exact same integer codes as the original encoding.
        In other words, the numerical values in the lagged features should 
        exactly match the integer codes that the `OrdinalEncoder` assigned.
        Formally, this means the following should hold true:

        `(X_train['lag_1'].cat.codes == X_train['lag_1']).all()`

        This consistency is guaranteed because:

        - `OrdinalEncoder` assigns integer codes starting from 0, in the alphabetical 
        order of category labels.

        - When autoregressive (lag) features are created later, they are converted 
        to pandas Categorical types using the same category ordering 
        (`categories = forecaster.class_codes_`).

        As a result, the categorical codes used in lag features remain aligned
        with the original encoding from the `OrdinalEncoder`.

        During prediction, we can work directly with NumPy arrays because the 
        `OrdinalEncoder` transforms new observations into the same integer codes 
        used by pandas Categorical during training. This eliminates the need to 
        convert data to pandas categorical types at inference time.
        
        """

        output = self._create_train_X_y(y=y, exog=exog, store_last_window=False)

        X_train = output[0]
        y_train = output[1]

        if not encoded:
            
            for col in self.lags_names:
                X_train[col] = self.encoder.inverse_transform(
                    X_train[col].to_numpy().reshape(-1, 1)
                ).ravel()

            y_train = pd.Series(
                          data  = self.encoder.inverse_transform(y_train.to_numpy().reshape(-1, 1)).ravel(),
                          index = y_train.index,
                          name  = y_train.name
                      )

        return X_train, y_train

    def _train_test_split_one_step_ahead(
        self,
        y: pd.Series,
        initial_train_size: int,
        exog: pd.Series | pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Create matrices needed to train and test the forecaster for one-step-ahead
        predictions.

        Parameters
        ----------
        y : pandas Series
            Training time series.
        initial_train_size : int
            Initial size of the training set. It is the number of observations used
            to train the forecaster before making the first prediction.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned.
        
        Returns
        -------
        X_train : pandas DataFrame
            Predictor values used to train the model.
        y_train : pandas Series
            Target values related to each row of `X_train`.
        X_test : pandas DataFrame
            Predictor values used to test the model.
        y_test : pandas Series
            Target values related to each row of `X_test`.
        
        """

        is_fitted = self.is_fitted
        self.is_fitted = False
        X_train, y_train, *_ = self._create_train_X_y(
            y    = y.iloc[: initial_train_size],
            exog = exog.iloc[: initial_train_size] if exog is not None else None
        )

        test_init = initial_train_size - self.window_size
        self.is_fitted = True
        X_test, y_test, *_ = self._create_train_X_y(
            y    = y.iloc[test_init:],
            exog = exog.iloc[test_init:] if exog is not None else None
        )

        self.is_fitted = is_fitted

        return X_train, y_train, X_test, y_test


    def create_sample_weights(
        self,
        X_train: pd.DataFrame,
    ) -> np.ndarray:
        """
        Create weights for each observation according to the forecaster's attribute
        `weight_func`.

        Parameters
        ----------
        X_train : pandas DataFrame
            Dataframe created with the `create_train_X_y` method, first return.

        Returns
        -------
        sample_weight : numpy ndarray
            Weights to use in `fit` method.

        """

        sample_weight = None

        if self.weight_func is not None:
            sample_weight = self.weight_func(X_train.index)

        if sample_weight is not None:
            if np.isnan(sample_weight).any():
                raise ValueError(
                    "The resulting `sample_weight` cannot have NaN values."
                )
            if np.any(sample_weight < 0):
                raise ValueError(
                    "The resulting `sample_weight` cannot have negative values."
                )
            if np.sum(sample_weight) == 0:
                raise ValueError(
                    "The resulting `sample_weight` cannot be normalized because "
                    "the sum of the weights is zero."
                )

        return sample_weight


    def fit(
        self,
        y: pd.Series,
        exog: pd.Series | pd.DataFrame | None = None,
        store_last_window: bool = True,
        store_in_sample_residuals: Any = None
    ) -> None:
        """
        Training Forecaster.

        Additional arguments to be passed to the `fit` method of the classifier 
        can be added with the `fit_kwargs` argument when initializing the forecaster.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned so
            that y[i] is regressed on exog[i].
        store_last_window : bool, default True
            Whether or not to store the last window (`last_window_`) of training data.
        store_in_sample_residuals : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        None

        Notes
        -----
        Categorical features are transformed using an `OrdinalEncoder` (self.encoder).
        The encoder's learned mappings (self.encoding_mapping_) are stored so that 
        later, when creating lag (autoregressive) features, the same category-to-integer 
        relationships can be applied consistently.

        The goal is to ensure that the lag features — which are recreated as 
        categorical variables — use the exact same integer codes as the original encoding.
        In other words, the numerical values in the lagged features should 
        exactly match the integer codes that the `OrdinalEncoder` assigned.
        Formally, this means the following should hold true:

        `(X_train['lag_1'].cat.codes == X_train['lag_1']).all()`

        This consistency is guaranteed because:

        - `OrdinalEncoder` assigns integer codes starting from 0, in the alphabetical 
        order of category labels.

        - When autoregressive (lag) features are created later, they are converted 
        to pandas Categorical types using the same category ordering 
        (`categories = forecaster.class_codes_`).

        As a result, the categorical codes used in lag features remain aligned
        with the original encoding from the `OrdinalEncoder`.

        During prediction, we can work directly with NumPy arrays because the 
        `OrdinalEncoder` transforms new observations into the same integer codes 
        used by pandas Categorical during training. This eliminates the need to 
        convert data to pandas categorical types at inference time.
        
        """

        # TODO: View which arguments to reset
        self.last_window_                       = None
        self.index_type_                        = None
        self.index_freq_                        = None
        self.training_range_                    = None
        self.series_name_in_                    = None
        self.exog_in_                           = False
        self.exog_names_in_                     = None
        self.exog_type_in_                      = None
        self.exog_dtypes_in_                    = None
        self.exog_dtypes_out_                   = None
        self.X_train_window_features_names_out_ = None
        self.X_train_exog_names_out_            = None
        self.X_train_features_names_out_        = None
        self.is_fitted                          = False
        self.fit_date                           = None
        self.classes_                           = None
        self.class_codes_                       = None
        self.n_classes_                         = None
        self.encoding_mapping_                  = None
        self.code_to_class_mapping_             = None

        (
            X_train,
            y_train,
            y_encoding_info_,
            exog_names_in_,
            X_train_window_features_names_out_,
            X_train_exog_names_out_,
            X_train_features_names_out_,
            exog_dtypes_in_,
            exog_dtypes_out_,
            last_window_
        ) = self._create_train_X_y(
                y=y, exog=exog, store_last_window=store_last_window
            )
        
        sample_weight = self.create_sample_weights(X_train=X_train)

        if sample_weight is not None:
            self.regressor.fit(
                X             = X_train,
                y             = y_train,
                sample_weight = sample_weight,
                **self.fit_kwargs
            )
        else:
            self.regressor.fit(X=X_train, y=y_train, **self.fit_kwargs)

        self.classes_ = y_encoding_info_['classes_']
        self.class_codes_ = y_encoding_info_['class_codes_']
        self.n_classes_ = y_encoding_info_['n_classes_']
        self.encoding_mapping_ = y_encoding_info_['encoding_mapping_']
        self.code_to_class_mapping_ = {
            code: cls for cls, code in self.encoding_mapping_.items()
        }

        self.X_train_window_features_names_out_ = X_train_window_features_names_out_
        self.X_train_features_names_out_ = X_train_features_names_out_

        self.is_fitted = True
        self.series_name_in_ = y.name if y.name is not None else 'y'
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range_ = y.index[[0, -1]]
        self.index_type_ = type(y.index)
        if isinstance(y.index, pd.DatetimeIndex):
            self.index_freq_ = y.index.freq
        else: 
            self.index_freq_ = y.index.step

        if exog is not None:
            self.exog_in_ = True
            self.exog_type_in_ = type(exog)
            self.exog_names_in_ = exog_names_in_
            self.exog_dtypes_in_ = exog_dtypes_in_
            self.exog_dtypes_out_ = exog_dtypes_out_
            self.X_train_exog_names_out_ = X_train_exog_names_out_

        if store_last_window:
            self.last_window_ = last_window_
        
    def _create_predict_inputs(
        self,
        steps: int | str | pd.Timestamp, 
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        check_inputs: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None, pd.Index, int]:
        """
        Create the inputs needed for the first iteration of the prediction 
        process. As this is a recursive process, the last window is updated at 
        each iteration of the prediction process.
        
        Parameters
        ----------
        steps : int, str, pandas Timestamp
            Number of steps to predict. 
            
            - If steps is int, number of steps to predict. 
            - If str or pandas Datetime, the prediction will be up to that date.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        check_inputs : bool, default True
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.

        Returns
        -------
        last_window_values : numpy ndarray
            Series values used to create the predictors needed in the first 
            iteration of the prediction (t + 1).
        exog_values : numpy ndarray, None
            Exogenous variable/s included as predictor/s.
        prediction_index : pandas Index
            Index of the predictions.
        steps: int
            Number of future steps predicted.
        
        """

        if last_window is None:
            last_window = self.last_window_

        if self.is_fitted:
            steps = date_to_index_position(
                        index        = last_window.index,
                        date_input   = steps,
                        method       = 'prediction',
                        date_literal = 'steps'
                    )

        if check_inputs:
            check_predict_input(
                forecaster_name = type(self).__name__,
                steps           = steps,
                is_fitted       = self.is_fitted,
                exog_in_        = self.exog_in_,
                index_type_     = self.index_type_,
                index_freq_     = self.index_freq_,
                window_size     = self.window_size,
                last_window     = last_window,
                exog            = exog,
                exog_names_in_  = self.exog_names_in_,
                interval        = None
            )

        # NOTE: NaNs are checked in check_predict_input, it creates a warning if found.
        last_window_values = (
            last_window.iloc[-self.window_size:].to_numpy(copy=True).ravel()
        )

        valid_classes = set(self.encoding_mapping_.keys())
        unique_values = set(last_window_values)
        invalid_values = unique_values - valid_classes
        
        if invalid_values:
            invalid_list = sorted(list(invalid_values))[:5]
            valid_list = sorted(list(valid_classes))[:10]
            
            raise ValueError(
                f"The `last_window` contains {len(invalid_values)} class label(s) "
                f"not seen during training: {invalid_list}{'...' if len(invalid_values) > 5 else ''}.\n"
                f"Valid class labels (seen during training): {valid_list}"
                f"{'...' if len(valid_classes) > 10 else ''}.\n"
                f"Total valid classes: {len(valid_classes)}."
            )

        # NOTE: Transform class labels to encoded values (same encoding used in 
        # training). This ensures that lag features will have the same numerical 
        # representation as during training.
        last_window_values = self.encoder.transform(
            last_window_values.reshape(-1, 1)
        ).ravel()

        if exog is not None:

            exog = input_to_frame(data=exog, input_name='exog')
            if exog.columns.tolist() != self.exog_names_in_:
                exog = exog[self.exog_names_in_]

            exog = transform_dataframe(
                       df                = exog,
                       transformer       = self.transformer_exog,
                       fit               = False,
                       inverse_transform = False
                   )
            
            # NOTE: Only check dtypes if they are not the same as seen in training
            if not exog.dtypes.to_dict() == self.exog_dtypes_out_:
                check_exog_dtypes(exog=exog)
            else:
                check_exog(exog=exog, allow_nan=False)
            
            exog_values = exog.to_numpy()[:steps]
        else:
            exog_values = None

        prediction_index = expand_index(
                               index = last_window.index,
                               steps = steps,
                           )

        return last_window_values, exog_values, prediction_index, steps


    def _recursive_predict(
        self,
        steps: int,
        last_window_values: np.ndarray,
        exog_values: np.ndarray | None = None,
        predict_proba: bool = False
    ) -> np.ndarray:
        """
        Predict n steps ahead. It is an iterative process in which, each prediction,
        is used as a predictor for the next step.
        
        Parameters
        ----------
        steps : int
            Number of steps to predict. 
        last_window_values : numpy ndarray
            Series values used to create the predictors needed in the first 
            iteration of the prediction (t + 1).
        exog_values : numpy ndarray, default None
            Exogenous variable/s included as predictor/s.
        predict_proba : bool, default False
            Whether to predict class probabilities instead of class labels.
        
        Returns
        -------
        predictions : numpy ndarray
            Predicted values if `predict_proba=False`, probability matrix of 
            shape (steps, n_classes) with the predicted probabilities for each class 
            at each step if `predict_proba=True`.

        """

        original_device = set_cpu_gpu_device(regressor=self.regressor, device='cpu')

        n_lags = len(self.lags) if self.lags is not None else 0
        n_window_features = (
            len(self.X_train_window_features_names_out_)
            if self.window_features is not None
            else 0
        )
        n_exog = exog_values.shape[1] if exog_values is not None else 0

        X = np.full(
            shape=(n_lags + n_window_features + n_exog), fill_value=np.nan, dtype=float
        )
        predictions = np.full(shape=steps, fill_value=np.nan, dtype=float)
        last_window = np.concatenate((last_window_values, predictions))

        if predict_proba:
            predictions = np.full(
                shape=(steps, self.n_classes_), fill_value=np.nan, dtype=float
            )

        for i in range(steps):

            if self.lags is not None:
                X[:n_lags] = last_window[-self.lags - (steps - i)]
            if self.window_features is not None:
                X[n_lags : n_lags + n_window_features] = np.concatenate(
                    [
                        wf.transform(last_window[i : -(steps - i)])
                        for wf in self.window_features
                    ]
                )
            if exog_values is not None:
                X[n_lags + n_window_features:] = exog_values[i]

            if predict_proba:
                proba = self.regressor.predict_proba(X.reshape(1, -1)).ravel()
                predictions[i, :] = proba
                pred = self.class_codes_[np.argmax(proba)]
            else:
                pred = self.regressor.predict(X.reshape(1, -1)).ravel().item()
                predictions[i] = pred

            # Update `last_window` values. The first position is discarded and 
            # the new prediction is added at the end.
            last_window[-(steps - i)] = pred

        set_cpu_gpu_device(regressor=self.regressor, device=original_device)

        return predictions

    # TODO: Adapt
    def create_predict_X(
        self,
        steps: int,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        check_inputs: bool = True
    ) -> pd.DataFrame:
        """
        Create the predictors needed to predict `steps` ahead. As it is a recursive
        process, the predictors are created at each iteration of the prediction 
        process.
        
        Parameters
        ----------
        steps : int, str, pandas Timestamp
            Number of steps to predict. 
            
            - If steps is int, number of steps to predict. 
            - If str or pandas Datetime, the prediction will be up to that date.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        check_inputs : bool, default True
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.

        Returns
        -------
        X_predict : pandas DataFrame
            Pandas DataFrame with the predictors for each step. The index 
            is the same as the prediction index.
        
        """

        (
            last_window_values,
            exog_values,
            prediction_index,
            steps
        ) = self._create_predict_inputs(
                steps        = steps,
                last_window  = last_window,
                exog         = exog,
                check_inputs = check_inputs,
            )
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="X does not have valid feature names", 
                category=UserWarning
            )
            predictions = self._recursive_predict(
                              steps              = steps,
                              last_window_values = last_window_values,
                              exog_values        = exog_values,
                              predict_proba      = False
                          )

        X_predict = []
        full_predictors = np.concatenate((last_window_values, predictions))

        if self.lags is not None:
            idx = np.arange(-steps, 0)[:, None] - self.lags
            X_lags = full_predictors[idx + len(full_predictors)]
            X_predict.append(X_lags)

        if self.window_features is not None:
            X_window_features = np.full(
                shape      = (steps, len(self.X_train_window_features_names_out_)), 
                fill_value = np.nan, 
                order      = 'C',
                dtype      = float
            )
            for i in range(steps):
                X_window_features[i, :] = np.concatenate(
                    [wf.transform(full_predictors[i:-(steps - i)]) 
                     for wf in self.window_features]
                )
            X_predict.append(X_window_features)

        if exog is not None:
            X_predict.append(exog_values)

        X_predict = pd.DataFrame(
                        data    = np.concatenate(X_predict, axis=1),
                        columns = self.X_train_features_names_out_,
                        index   = prediction_index
                    )
        
        if self.use_native_categoricals:
            for col in self.lags_names:
                X_predict[col] = pd.Categorical(
                                     values     = X_predict[col],
                                     categories = self.class_codes_,
                                     ordered    = False
                                 )
        
        if self.exog_in_:
            categorical_features = any(
                not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) 
                for dtype in set(self.exog_dtypes_out_)
            )
            if categorical_features:
                X_predict = X_predict.astype(self.exog_dtypes_out_)

        if self.transformer_exog is not None:
            warnings.warn(
                "The output matrix is in the transformed scale due to the "
                "inclusion of transformations (`transformer_exog`) in the Forecaster. "
                "As a result, any predictions generated using this matrix will also "
                "be in the transformed scale. Please refer to the documentation "
                "for more details: "
                "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html",
                DataTransformationWarning
            )

        return X_predict

    def predict(
        self,
        steps: int | str | pd.Timestamp,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        check_inputs: bool = True
    ) -> pd.Series:
        """
        Predict n steps ahead. It is an recursive process in which, each prediction,
        is used as a predictor for the next step.
        
        Parameters
        ----------
        steps : int, str, pandas Timestamp
            Number of steps to predict. 
            
            - If steps is int, number of steps to predict. 
            - If str or pandas Datetime, the prediction will be up to that date.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        check_inputs : bool, default True
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.

        Returns
        -------
        predictions : pandas Series
            Predicted values (class labels).
        
        """

        (
            last_window_values,
            exog_values,
            prediction_index,
            steps
        ) = self._create_predict_inputs(
                steps        = steps,
                last_window  = last_window,
                exog         = exog,
                check_inputs = check_inputs
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="X does not have valid feature names", 
                category=UserWarning
            )
            predictions = self._recursive_predict(
                              steps              = steps,
                              last_window_values = last_window_values,
                              exog_values        = exog_values,
                              predict_proba      = False
                          )

        predictions = self.encoder.inverse_transform(
            predictions.reshape(-1, 1)
        ).ravel()

        predictions = pd.Series(
                          data  = predictions,
                          index = prediction_index,
                          name  = 'pred'
                      )

        return predictions
    
    def predict_proba(
        self,
        steps: int | str | pd.Timestamp,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        check_inputs: bool = True
    ) -> pd.DataFrame:
        """
        Predict class probabilities n steps ahead. It is a recursive process in 
        which the predicted class (argmax of probabilities) is used as a predictor 
        for the next step.
        
        Parameters
        ----------
        steps : int, str, pandas Timestamp
            Number of steps to predict.
            
            - If steps is int, number of steps to predict. 
            - If str or pandas Datetime, the prediction will be up to that date.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        check_inputs : bool, default True
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.
        
        Returns
        -------
        probabilities : pandas DataFrame
            Predicted probabilities for each class. Shape (steps, n_classes).
            Columns are the original class labels.
        
        """

        if not hasattr(self.regressor, 'predict_proba'):
            raise AttributeError(
                f"The classifier {type(self.regressor).__name__} does not have a "
                f"`predict_proba` method. Use a classifier that supports probability "
                f"predictions (e.g., XGBClassifier, HistGradientBoostingClassifier, etc.)."
            )
        
        (
            last_window_values,
            exog_values,
            prediction_index,
            steps
        ) = self._create_predict_inputs(
                steps        = steps,
                last_window  = last_window,
                exog         = exog,
                check_inputs = check_inputs
            )
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="X does not have valid feature names", 
                category=UserWarning
            )
            probabilities = self._recursive_predict(
                                steps              = steps,
                                last_window_values = last_window_values,
                                exog_values        = exog_values,
                                predict_proba      = True
                            )
        
        probabilities = pd.DataFrame(
                            data    = probabilities,
                            index   = prediction_index,
                            columns = [f"{cls}_proba" for cls in self.classes_]
                        )
        
        return probabilities

    def set_params(
        self, 
        params: dict[str, object]
    ) -> None:
        """
        Set new values to the parameters of the scikit-learn model stored in the
        forecaster.
        
        Parameters
        ----------
        params : dict
            Parameters values.

        Returns
        -------
        None
        
        """

        self.regressor = clone(self.regressor)
        self.regressor.set_params(**params)

    def set_fit_kwargs(
        self, 
        fit_kwargs: dict[str, object]
    ) -> None:
        """
        Set new values for the additional keyword arguments passed to the `fit` 
        method of the classifier.
        
        Parameters
        ----------
        fit_kwargs : dict
            Dict of the form {"argument": new_value}.

        Returns
        -------
        None
        
        """

        self.fit_kwargs = check_select_fit_kwargs(self.regressor, fit_kwargs=fit_kwargs)

    def set_lags(
        self, 
        lags: int | list[int] | np.ndarray[int] | range[int] | None = None
    ) -> None:
        """
        Set new value to the attribute `lags`. Attributes `lags_names`, 
        `max_lag` and `window_size` are also updated.
        
        Parameters
        ----------
        lags : int, list, numpy ndarray, range, default None
            Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1. 
        
            - `int`: include lags from 1 to `lags` (included).
            - `list`, `1d numpy ndarray` or `range`: include only lags present in 
            `lags`, all elements must be int.
            - `None`: no lags are included as predictors. 

        Returns
        -------
        None
        
        """

        if self.window_features is None and lags is None:
            raise ValueError(
                "At least one of the arguments `lags` or `window_features` "
                "must be different from None. This is required to create the "
                "predictors used in training the forecaster."
            )
        
        self.lags, self.lags_names, self.max_lag = initialize_lags(type(self).__name__, lags)
        self.window_size = max(
            [ws for ws in [self.max_lag, self.max_size_window_features] 
             if ws is not None]
        )

    def set_window_features(
        self, 
        window_features: object | list[object] | None = None
    ) -> None:
        """
        Set new value to the attribute `window_features`. Attributes 
        `max_size_window_features`, `window_features_names`, 
        `window_features_class_names` and `window_size` are also updated.
        
        Parameters
        ----------
        window_features : object, list, default None
            Instance or list of instances used to create window features. Window features
            are created from the original time series and are included as predictors.

        Returns
        -------
        None
        
        """

        if window_features is None and self.lags is None:
            raise ValueError(
                "At least one of the arguments `lags` or `window_features` "
                "must be different from None. This is required to create the "
                "predictors used in training the forecaster."
            )
        
        self.window_features, self.window_features_names, self.max_size_window_features = (
            initialize_window_features(window_features)
        )
        self.window_features_class_names = None
        if window_features is not None:
            self.window_features_class_names = [
                type(wf).__name__ for wf in self.window_features
            ] 
        self.window_size = max(
            [ws for ws in [self.max_lag, self.max_size_window_features] 
             if ws is not None]
        )

    def get_feature_importances(
        self,
        sort_importance: bool = True
    ) -> pd.DataFrame:
        """
        Return feature importances of the classifier stored in the forecaster.
        Only valid when classifier stores internally the feature importances in the
        attribute `feature_importances_` or `coef_`. Otherwise, returns `None`.

        Parameters
        ----------
        sort_importance: bool, default True
            If `True`, sorts the feature importances in descending order.

        Returns
        -------
        feature_importances : pandas DataFrame
            Feature importances associated with each predictor.

        """

        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `get_feature_importances()`."
            )

        if isinstance(self.regressor, Pipeline):
            estimator = self.regressor[-1]
        else:
            estimator = self.regressor

        if hasattr(estimator, 'feature_importances_'):
            feature_importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            feature_importances = estimator.coef_
        else:
            warnings.warn(
                f"Impossible to access feature importances for classifier of type "
                f"{type(estimator)}. This method is only valid when the "
                f"classifier stores internally the feature importances in the "
                f"attribute `feature_importances_` or `coef_`."
            )
            feature_importances = None

        if feature_importances is not None:
            feature_importances = pd.DataFrame({
                                      'feature': self.X_train_features_names_out_,
                                      'importance': feature_importances
                                  })
            if sort_importance:
                feature_importances = feature_importances.sort_values(
                                          by='importance', ascending=False
                                      )

        return feature_importances
