################################################################################
#                     skforecast.model_selection._split                        #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from copy import deepcopy
from typing import Union, Optional, Any
import warnings
import numpy as np
import pandas as pd
import itertools
from ..exceptions import IgnoredArgumentWarning


class BaseFold():
    """
    Base class for all Fold classes in skforecast. All fold classes should specify
    all the parameters that can be set at the class level in their ``__init__``.

    Parameters
    ----------
    steps : int, default `None`
        Number of observations used to be predicted in each fold. This is also commonly
        referred to as the forecast horizon or test size.
    initial_train_size : int, default `None`
        Number of observations used for initial training.
    window_size : int, default `None`
        Number of observations needed to generate the autoregressive predictors.
    differentiation : int, default `None`
        Number of observations to use for differentiation. This is used to extend the
        `last_window` as many observations as the differentiation order.
    refit : bool, int, default `False`
        Whether to refit the forecaster in each fold.

        - If `True`, the forecaster is refitted in each fold.
        - If `False`, the forecaster is trained only in the first fold.
        - If an integer, the forecaster is trained in the first fold and then refitted
          every `refit` folds.
    fixed_train_size : bool, default `True`
        Whether the training size is fixed or increases in each fold.
    gap : int, default `0`
        Number of observations between the end of the training set and the start of the
        test set.
    skip_folds : int, list, default `None`
        Number of folds to skip.

        - If an integer, every 'skip_folds'-th is returned.
        - If a list, the indexes of the folds to skip.

        For example, if `skip_folds=3` and there are 10 folds, the returned folds are
        0, 3, 6, and 9. If `skip_folds=[1, 2, 3]`, the returned folds are 0, 4, 5, 6, 7,
        8, and 9.
    allow_incomplete_fold : bool, default `True`
        Whether to allow the last fold to include fewer observations than `steps`.
        If `False`, the last fold is excluded if it is incomplete.
    return_all_indexes : bool, default `False`
        Whether to return all indexes or only the start and end indexes of each fold.
    verbose : bool, default `True`
        Whether to print information about generated folds.

    Attributes
    ----------
    steps : int
        Number of observations used to be predicted in each fold. This is also commonly
        referred to as the forecast horizon or test size.
    initial_train_size : int
        Number of observations used for initial training.
    window_size : int
        Number of observations needed to generate the autoregressive predictors.
    differentiation : int
        Number of observations to use for differentiation. This is used to extend the
        `last_window` as many observations as the differentiation order.
    refit : bool, int
        Whether to refit the forecaster in each fold.
    fixed_train_size : bool
        Whether the training size is fixed or increases in each fold.
    gap : int
        Number of observations between the end of the training set and the start of the
        test set.
    skip_folds : int, list
        Number of folds to skip.
    allow_incomplete_fold : bool
        Whether to allow the last fold to include fewer observations than `steps`.
    return_all_indexes : bool
        Whether to return all indexes or only the start and end indexes of each fold.
    verbose : bool
        Whether to print information about generated folds.

    """

    def __init__(
        self,
        steps: Optional[int] = None,
        initial_train_size: Optional[int] = None,
        window_size: Optional[int] = None,
        differentiation: Optional[int] = None,
        refit: Union[bool, int] = False,
        fixed_train_size: bool = True,
        gap: int = 0,
        skip_folds: Optional[Union[int, list]] = None,
        allow_incomplete_fold: bool = True,
        return_all_indexes: bool = False,
        verbose: bool = True
    ) -> None:

        self._validate_params(
            cv_name               = type(self).__name__,
            steps                 = steps,
            initial_train_size    = initial_train_size,
            window_size           = window_size,
            differentiation       = differentiation,
            refit                 = refit,
            fixed_train_size      = fixed_train_size,
            gap                   = gap,
            skip_folds            = skip_folds,
            allow_incomplete_fold = allow_incomplete_fold,
            return_all_indexes    = return_all_indexes,
            verbose               = verbose
        )

        self.steps                 = steps
        self.initial_train_size    = initial_train_size
        self.window_size           = window_size
        self.differentiation       = differentiation
        self.refit                 = refit
        self.fixed_train_size      = fixed_train_size
        self.gap                   = gap
        self.skip_folds            = skip_folds
        self.allow_incomplete_fold = allow_incomplete_fold
        self.return_all_indexes    = return_all_indexes
        self.verbose               = verbose

    def _validate_params(
        self,
        cv_name: str,
        steps: Optional[int] = None,
        initial_train_size: Optional[int] = None,
        window_size: Optional[int] = None,
        differentiation: Optional[int] = None,
        refit: Union[bool, int] = False,
        fixed_train_size: bool = True,
        gap: int = 0,
        skip_folds: Optional[Union[int, list]] = None,
        allow_incomplete_fold: bool = True,
        return_all_indexes: bool = False,
        verbose: bool = True
    ) -> None: 
        """
        Validate all input parameters to ensure correctness.
        """

        if cv_name == "TimeSeriesFold":
            if not isinstance(steps, (int, np.integer)) or steps < 1:
                raise ValueError(
                    f"`steps` must be an integer greater than 0. Got {steps}."
                )
            if not isinstance(initial_train_size, (int, np.integer, type(None))):
                raise ValueError(
                    f"`initial_train_size` must be an integer greater than 0 or None. "
                    f"Got {initial_train_size}."
                )
            if initial_train_size is not None and initial_train_size < 1:
                raise ValueError(
                    f"`initial_train_size` must be an integer greater than 0 or None. "
                    f"Got {initial_train_size}."
                )
            if not isinstance(refit, (bool, int, np.integer)):
                raise TypeError(
                    f"`refit` must be a boolean or an integer equal or greater than 0. "
                    f"Got {refit}."
                )
            if isinstance(refit, (int, np.integer)) and not isinstance(refit, bool) and refit < 0:
                raise TypeError(
                    f"`refit` must be a boolean or an integer equal or greater than 0. "
                    f"Got {refit}."
                )
            if not isinstance(fixed_train_size, bool):
                raise TypeError(
                    (f"`fixed_train_size` must be a boolean: `True`, `False`. "
                     f"Got {fixed_train_size}.")
                )
            if not isinstance(gap, (int, np.integer)) or gap < 0:
                raise ValueError(
                    f"`gap` must be an integer greater than or equal to 0. Got {gap}."
                )
            if skip_folds is not None:
                if not isinstance(skip_folds, (int, np.integer, list, type(None))):
                    raise TypeError(
                        (f"`skip_folds` must be an integer greater than 0, a list of "
                         f"integers or `None`. Got {skip_folds}.")
                    )
                if isinstance(skip_folds, (int, np.integer)) and skip_folds < 1:
                    raise ValueError(
                        (f"`skip_folds` must be an integer greater than 0, a list of "
                         f"integers or `None`. Got {skip_folds}.")
                    )
                if isinstance(skip_folds, list) and any([x < 1 for x in skip_folds]):
                    raise ValueError(
                        (f"`skip_folds` list must contain integers greater than or "
                         f"equal to 1. The first fold is always needed to train the "
                        f"forecaster. Got {skip_folds}.")
                    ) 
            if not isinstance(allow_incomplete_fold, bool):
                raise TypeError(
                    (f"`allow_incomplete_fold` must be a boolean: `True`, `False`. "
                    f"Got {allow_incomplete_fold}.")
                )
            
        if cv_name == "OneStepAheadFold":
            if (
                not isinstance(initial_train_size, (int, np.integer))
                or initial_train_size < 1
            ):
                raise ValueError(
                    f"`initial_train_size` must be an integer greater than 0. "
                    f"Got {initial_train_size}."
                )
        
        if (
            not isinstance(window_size, (int, np.integer, pd.DateOffset, type(None)))
            or isinstance(window_size, (int, np.integer))
            and window_size < 1
        ):
            raise ValueError(
                f"`window_size` must be an integer greater than 0. Got {window_size}."
            )
        
        if not isinstance(return_all_indexes, bool):
            raise TypeError(
                (f"`return_all_indexes` must be a boolean: `True`, `False`. "
                 f"Got {return_all_indexes}.")
            )
        if differentiation is not None:
            if not isinstance(differentiation, (int, np.integer)) or differentiation < 0:
                raise ValueError(
                    (f"`differentiation` must be None or an integer greater than or "
                     f"equal to 0. Got {differentiation}.")
                )
        if not isinstance(verbose, bool):
            raise TypeError(
                (f"`verbose` must be a boolean: `True`, `False`. "
                 f"Got {verbose}.")
            )

    def _extract_index(
        self,
        X: Union[pd.Series, pd.DataFrame, pd.Index, dict]
    ) -> pd.Index:
        """
        Extracts and returns the index from the input data X.

        Parameters
        ----------
        X : pandas Series, pandas DataFrame, pandas Index, dict
            Time series data or index to split.

        Returns
        -------
        idx : pandas Index
            Index extracted from the input data.
        
        """

        if isinstance(X, (pd.Series, pd.DataFrame)):
            idx = X.index
        elif isinstance(X, dict):
            freqs = [s.index.freq for s in X.values() if s.index.freq is not None]
            if not freqs:
                raise ValueError("At least one series must have a frequency.")
            if not all(f == freqs[0] for f in freqs):
                raise ValueError(
                    "All series with frequency must have the same frequency."
                )
            min_idx = min([v.index[0] for v in X.values()])
            max_idx = max([v.index[-1] for v in X.values()])
            idx = pd.date_range(start=min_idx, end=max_idx, freq=freqs[0])
        else:
            idx = X
            
        return idx

    def set_params(
        self, 
        params: dict
    ) -> None:
        """
        Set the parameters of the TimeSeriesFold object. Before overwriting the
        current parameters, the input parameters are validated to ensure correctness.

        Parameters
        ----------
        params : dict
            Dictionary with the parameters to set.
        
        Returns
        -------
        None
        
        """

        if not isinstance(params, dict):
            raise ValueError(
                f"`params` must be a dictionary. Got {type(params)}."
            )

        current_params = deepcopy(vars(self))
        unknown_params = set(params.keys()) - set(current_params.keys())
        if unknown_params:
            warnings.warn(
                f"Unknown parameters: {unknown_params}. They have been ignored.",
                IgnoredArgumentWarning
            )
        updated_params = {'cv_name': type(self).__name__, **current_params, **params}
        self._validate_params(**updated_params)
        for key, value in updated_params.items():
            setattr(self, key, value)


class OneStepAheadFold(BaseFold):
    """
    Class to split time series data into train and test folds for one-step-ahead
    forecasting.

    Parameters
    ----------
    initial_train_size : int
        Number of observations used for initial training.
    window_size : int, default `None`
        Number of observations needed to generate the autoregressive predictors.
    differentiation : int, default `None`
        Number of observations to use for differentiation. This is used to extend the
        `last_window` as many observations as the differentiation order.
    return_all_indexes : bool, default `False`
        Whether to return all indexes or only the start and end indexes of each fold.
    verbose : bool, default `True`
        Whether to print information about generated folds.

    Attributes
    ----------
    initial_train_size : int
        Number of observations used for initial training.
    window_size : int
        Number of observations needed to generate the autoregressive predictors.
    differentiation : int 
        Number of observations to use for differentiation. This is used to extend the
        `last_window` as many observations as the differentiation order.
    return_all_indexes : bool
        Whether to return all indexes or only the start and end indexes of each fold.
    verbose : bool
        Whether to print information about generated folds.
    steps : Any
        This attribute is not used in this class. It is included for API consistency.
    fixed_train_size : Any
        This attribute is not used in this class. It is included for API consistency.
    gap : Any
        This attribute is not used in this class. It is included for API consistency.
    skip_folds : Any
        This attribute is not used in this class. It is included for API consistency.
    allow_incomplete_fold : Any
        This attribute is not used in this class. It is included for API consistency.
    refit : Any
        This attribute is not used in this class. It is included for API consistency.
    
    """

    def __init__(
        self,
        initial_train_size: int,
        window_size: Optional[int] = None,
        differentiation: Optional[int] = None,
        return_all_indexes: bool = False,
        verbose: bool = True,
    ) -> None:
        
        super().__init__(
            initial_train_size = initial_train_size,
            window_size        = window_size,
            differentiation    = differentiation,
            return_all_indexes = return_all_indexes,
            verbose            = verbose
        )

    def __repr__(
        self
    ) -> str:
        """
        Information displayed when printed.
        """
            
        return (
            f"OneStepAheadFold(\n"
            f"    initial_train_size = {self.initial_train_size},\n"
            f"    window_size        = {self.window_size},\n"
            f"    differentiation    = {self.differentiation},\n"
            f"    return_all_indexes = {self.return_all_indexes},\n"
            f"    verbose            = {self.verbose}\n"
            f")"
        )
    
    def split(
        self,
        X: Union[pd.Series, pd.DataFrame, pd.Index, dict],
        as_pandas: bool = False,
        externally_fitted: Any = None
    ) -> Union[list, pd.DataFrame]:
        """
        Split the time series data into train and test folds.

        Parameters
        ----------
        X : pandas Series, DataFrame, Index, or dictionary
            Time series data or index to split.
        as_pandas : bool, default `False`
            If True, the folds are returned as a DataFrame. This is useful to visualize
            the folds in a more interpretable way.
        externally_fitted : Any
            This argument is not used in this class. It is included for API consistency.
        
        Returns
        -------
        fold : list, pandas DataFrame
            A list of lists containing the indices (position) for for each fold. Each list
            contains 2 lists the following information:

            - [train_start, train_end]: list with the start and end positions of the
            training set.
            - [test_start, test_end]: list with the start and end positions of the test
            set. These are the observations used to evaluate the forecaster.
        
            It is important to note that the returned values are the positions of the
            observations and not the actual values of the index, so they can be used to
            slice the data directly using iloc.

            If `as_pandas` is `True`, the folds are returned as a DataFrame with the
            following columns: 'fold', 'train_start', 'train_end', 'test_start', 'test_end'.

            Following the python convention, the start index is inclusive and the end
            index is exclusive. This means that the last index is not included in the
            slice.
        
        """

        if not isinstance(X, (pd.Series, pd.DataFrame, pd.Index, dict)):
            raise TypeError(
                (f"X must be a pandas Series, DataFrame, Index or a dictionary. "
                 f"Got {type(X)}.")
            )

        index = self._extract_index(X)
        fold = [
            [0, self.initial_train_size],
            [self.initial_train_size, len(X)],
            True
        ]

        if self.verbose:
            self._print_info(
                index = index,
                fold = fold,
            )

        if self.return_all_indexes:
            fold = [
                [range(fold[0][0], fold[0][1])],
                [range(fold[1][0], fold[1][1])],
                fold[2]
            ]

        if as_pandas:
            if not self.return_all_indexes:
                fold = pd.DataFrame(
                    data = [list(itertools.chain(*fold[:-1])) + [fold[-1]]],
                    columns = [
                        'train_start',
                        'train_end',
                        'test_start',
                        'test_end',
                        'fit_forecaster'
                    ],
                )
            else:
                fold = pd.DataFrame(
                    data = [fold],
                    columns = [
                        'train_index',
                        'test_index',
                        'fit_forecaster'
                    ],
                )
            fold.insert(0, 'fold', range(len(fold)))

        return fold

    def _print_info(
        self,
        index: pd.Index,
        fold: list,
    ) -> None:
        """
        Print information about folds.
        """

        if self.differentiation is None:
            differentiation = 0
        else:
            differentiation = self.differentiation

        initial_train_size = self.initial_train_size - differentiation
        test_length = len(index) - (initial_train_size + differentiation)

        print("Information of folds")
        print("--------------------")
        print(
            f"Number of observations in train: {initial_train_size}"
        )
        if self.differentiation is not None:
            print(
                f"    First {differentiation} observation/s in training set "
                f"are used for differentiation"
            )
        print(
            f"Number of observations in test: {test_length}"
        )
        
        training_start = index[fold[0][0] + differentiation]
        training_end = index[fold[0][-1]]
        test_start  = index[fold[1][0]]
        test_end    = index[fold[1][-1] - 1]
        
        print(
            f"Training : {training_start} -- {training_end} (n={initial_train_size})"
        )
        print(
            f"Test     : {test_start} -- {test_end} (n={test_length})"
        )
        print("")


class TimeSeriesFold(BaseFold):
    """
    Class to split time series data into train and test folds. 
    When used within a backtesting or hyperparameter search, the arguments
    'initial_train_size', 'window_size' and 'differentiation' are not required
    as they are automatically set by the backtesting or hyperparameter search
    functions.

    Parameters
    ----------
    steps : int
        Number of observations used to be predicted in each fold. This is also commonly
        referred to as the forecast horizon or test size.
    initial_train_size : int, default `None`
        Number of observations used for initial training. If `None` or 0, the initial
        forecaster is not trained in the first fold.
    window_size : int, default `None`
        Number of observations needed to generate the autoregressive predictors.
    differentiation : int, default `None`
        Number of observations to use for differentiation. This is used to extend the
        `last_window` as many observations as the differentiation order.
    refit : bool, int, default `False`
        Whether to refit the forecaster in each fold.

        - If `True`, the forecaster is refitted in each fold.
        - If `False`, the forecaster is trained only in the first fold.
        - If an integer, the forecaster is trained in the first fold and then refitted
          every `refit` folds.
    fixed_train_size : bool, default `True`
        Whether the training size is fixed or increases in each fold.
    gap : int, default `0`
        Number of observations between the end of the training set and the start of the
        test set.
    skip_folds : int, list, default `None`
        Number of folds to skip.

        - If an integer, every 'skip_folds'-th is returned.
        - If a list, the indexes of the folds to skip.

        For example, if `skip_folds=3` and there are 10 folds, the returned folds are
        0, 3, 6, and 9. If `skip_folds=[1, 2, 3]`, the returned folds are 0, 4, 5, 6, 7,
        8, and 9.
    allow_incomplete_fold : bool, default `True`
        Whether to allow the last fold to include fewer observations than `steps`.
        If `False`, the last fold is excluded if it is incomplete.
    return_all_indexes : bool, default `False`
        Whether to return all indexes or only the start and end indexes of each fold.
    verbose : bool, default `True`
        Whether to print information about generated folds.

    Attributes
    ----------
    steps : int
        Number of observations used to be predicted in each fold. This is also commonly
        referred to as the forecast horizon or test size.
    initial_train_size : int
        Number of observations used for initial training. If `None` or 0, the initial
        forecaster is not trained in the first fold.
    window_size : int
        Number of observations needed to generate the autoregressive predictors.
    differentiation : int
        Number of observations to use for differentiation. This is used to extend the
        `last_window` as many observations as the differentiation order.
    refit : bool, int
        Whether to refit the forecaster in each fold.
    fixed_train_size : bool
        Whether the training size is fixed or increases in each fold.
    gap : int
        Number of observations between the end of the training set and the start of the
        test set.
    skip_folds : int, list
        Number of folds to skip.
    allow_incomplete_fold : bool
        Whether to allow the last fold to include fewer observations than `steps`.
    return_all_indexes : bool
        Whether to return all indexes or only the start and end indexes of each fold.
    verbose : bool
        Whether to print information about generated folds.

    Notes
    -----
    Returned values are the positions of the observations and not the actual values of
    the index, so they can be used to slice the data directly using iloc. For example,
    if the input series is `X = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`, the 
    `initial_train_size = 3`, `window_size = 2`, `steps = 4`, and `gap = 1`,
    the output of the first fold will: [[0, 3], [1, 3], [3, 8], [4, 8], True].

    The first list `[0, 3]` indicates that the training set goes from the first to the
    third observation. The second list `[1, 3]` indicates that the last window seen by
    the forecaster during training goes from the second to the third observation. The
    third list `[3, 8]` indicates that the test set goes from the fourth to the eighth
    observation. The fourth list `[4, 8]` indicates that the test set including the gap
    goes from the fifth to the eighth observation. The boolean `False` indicates that the
    forecaster should not be trained in this fold.

    Following the python convention, the start index is inclusive and the end index is
    exclusive. This means that the last index is not included in the slice.

    """

    def __init__(
        self,
        steps: int,
        initial_train_size: Optional[int] = None,
        window_size: Optional[int] = None,
        differentiation: Optional[int] = None,
        refit: Union[bool, int] = False,
        fixed_train_size: bool = True,
        gap: int = 0,
        skip_folds: Optional[Union[int, list]] = None,
        allow_incomplete_fold: bool = True,
        return_all_indexes: bool = False,
        verbose: bool = True
    ) -> None:
        
        super().__init__(
            steps                 = steps,
            initial_train_size    = initial_train_size,
            window_size           = window_size,
            differentiation       = differentiation,
            refit                 = refit,
            fixed_train_size      = fixed_train_size,
            gap                   = gap,
            skip_folds            = skip_folds,
            allow_incomplete_fold = allow_incomplete_fold,
            return_all_indexes    = return_all_indexes,
            verbose               = verbose
        )

    def __repr__(
        self
    ) -> str:
        """
        Information displayed when printed.
        """
            
        return (
            f"TimeSeriesFold(\n"
            f"    steps                 = {self.steps},\n"
            f"    initial_train_size    = {self.initial_train_size},\n"
            f"    window_size           = {self.window_size},\n"
            f"    differentiation       = {self.differentiation},\n"
            f"    refit                 = {self.refit},\n"
            f"    fixed_train_size      = {self.fixed_train_size},\n"
            f"    gap                   = {self.gap},\n"
            f"    skip_folds            = {self.skip_folds},\n"
            f"    allow_incomplete_fold = {self.allow_incomplete_fold},\n"
            f"    return_all_indexes    = {self.return_all_indexes},\n"
            f"    verbose               = {self.verbose}\n"
            f")"
        )

    def split(
        self,
        X: Union[pd.Series, pd.DataFrame, pd.Index, dict],
        as_pandas: bool = False
    ) -> Union[list, pd.DataFrame]:
        """
        Split the time series data into train and test folds.

        Parameters
        ----------
        X : pandas Series, pandas DataFrame, pandas Index, dict
            Time series data or index to split.
        as_pandas : bool, default `False`
            If True, the folds are returned as a DataFrame. This is useful to visualize
            the folds in a more interpretable way.

        Returns
        -------
        folds : list, pandas DataFrame
            A list of lists containing the indices (position) for for each fold. Each list
            contains 4 lists and a boolean with the following information:

            - [train_start, train_end]: list with the start and end positions of the
            training set.
            - [last_window_start, last_window_end]: list with the start and end positions
            of the last window seen by the forecaster during training. The last window
            is used to generate the lags use as predictors. If `differentiation` is
            included, the interval is extended as many observations as the
            differentiation order. If the argument `window_size` is `None`, this list is
            empty.
            - [test_start, test_end]: list with the start and end positions of the test
            set. These are the observations used to evaluate the forecaster.
            - [test_start_with_gap, test_end_with_gap]: list with the start and end
            positions of the test set including the gap. The gap is the number of
            observations between the end of the training set and the start of the test
            set.
            - fit_forecaster: boolean indicating whether the forecaster should be fitted
            in this fold.

            It is important to note that the returned values are the positions of the
            observations and not the actual values of the index, so they can be used to
            slice the data directly using iloc.

            If `as_pandas` is `True`, the folds are returned as a DataFrame with the
            following columns: 'fold', 'train_start', 'train_end', 'last_window_start',
            'last_window_end', 'test_start', 'test_end', 'test_start_with_gap',
            'test_end_with_gap', 'fit_forecaster'.

            Following the python convention, the start index is inclusive and the end
            index is exclusive. This means that the last index is not included in the
            slice.

        """

        if not isinstance(X, (pd.Series, pd.DataFrame, pd.Index, dict)):
            raise TypeError(
                (f"X must be a pandas Series, DataFrame, Index or a dictionary. "
                 f"Got {type(X)}.")
            )
        
        if isinstance(self.window_size, pd.tseries.offsets.DateOffset):
            # Calculate the window_size in steps. This is not a exact calculation
            # because the offset follows the calendar rules and the distance between
            # two dates may not be constant.
            first_valid_index = X.index[-1] - self.window_size
            try:
                window_size_idx_start = X.index.get_loc(first_valid_index)
                window_size_idx_end = X.index.get_loc(X.index[-1])
                self.window_size = window_size_idx_end - window_size_idx_start
            except KeyError:
                raise ValueError(
                    f"The length of `X` ({len(X)}), must be greater than or equal "
                    f"to the window size ({self.window_size}). Try to decrease the "
                    f"size of the offset (forecaster.offset), or increase the "
                    f"size of `y`."
                )
        
        if self.initial_train_size is None:
            if self.window_size is None:
                raise ValueError(
                    "To use split method when `initial_train_size` is None, "
                    "`window_size` must be an integer greater than 0. "
                    "Although no initial training is done and all data is used to "
                    "evaluate the model, the first `window_size` observations are "
                    "needed to create the initial predictors. Got `window_size` = None."
                )
            if self.refit:
                raise ValueError(
                    "`refit` is only allowed when `initial_train_size` is not `None`. "
                    "Set `refit` to `False` if you want to use `initial_train_size = None`."
                )
            externally_fitted = True
            self.initial_train_size = self.window_size  # Reset to None later
        else:
            if self.window_size is None:
                warnings.warn(
                    "Last window cannot be calculated because `window_size` is None."
                )
            externally_fitted = False

        index = self._extract_index(X)
        idx = range(len(index))
        folds = []
        i = 0
        last_fold_excluded = False

        if len(index) < self.initial_train_size + self.steps:
            raise ValueError(
                (f"The time series must have at least `initial_train_size + steps` "
                 f"observations. Got {len(index)} observations.")
            )

        while self.initial_train_size + (i * self.steps) + self.gap < len(index):

            if self.refit:
                # If `fixed_train_size` the train size doesn't increase but moves by 
                # `steps` positions in each iteration. If `False`, the train size
                # increases by `steps` in each iteration.
                train_iloc_start = i * (self.steps) if self.fixed_train_size else 0
                train_iloc_end = self.initial_train_size + i * (self.steps)
                test_iloc_start = train_iloc_end
            else:
                # The train size doesn't increase and doesn't move.
                train_iloc_start = 0
                train_iloc_end = self.initial_train_size
                test_iloc_start = self.initial_train_size + i * (self.steps)
            
            if self.window_size is not None:
                last_window_iloc_start = test_iloc_start - self.window_size
            test_iloc_end = test_iloc_start + self.gap + self.steps
        
            partitions = [
                idx[train_iloc_start : train_iloc_end],
                idx[last_window_iloc_start : test_iloc_start] if self.window_size is not None else [],
                idx[test_iloc_start : test_iloc_end],
                idx[test_iloc_start + self.gap : test_iloc_end]
            ]
            folds.append(partitions)
            i += 1

        if not self.allow_incomplete_fold and len(folds[-1][3]) < self.steps:
            folds = folds[:-1]
            last_fold_excluded = True

        # Replace partitions inside folds with length 0 with `None`
        folds = [
            [partition if len(partition) > 0 else None for partition in fold] 
             for fold in folds
        ]

        # Create a flag to know whether to train the forecaster
        if self.refit == 0:
            self.refit = False
            
        if isinstance(self.refit, bool):
            fit_forecaster = [self.refit] * len(folds)
            fit_forecaster[0] = True
        else:
            fit_forecaster = [False] * len(folds)
            for i in range(0, len(fit_forecaster), self.refit): 
                fit_forecaster[i] = True
        
        for i in range(len(folds)): 
            folds[i].append(fit_forecaster[i])
            if fit_forecaster[i] is False:
                folds[i][0] = folds[i - 1][0]

        index_to_skip = []
        if self.skip_folds is not None:
            if isinstance(self.skip_folds, (int, np.integer)) and self.skip_folds > 0:
                index_to_keep = np.arange(0, len(folds), self.skip_folds)
                index_to_skip = np.setdiff1d(np.arange(0, len(folds)), index_to_keep, assume_unique=True)
                index_to_skip = [int(x) for x in index_to_skip]  # Required since numpy 2.0
            if isinstance(self.skip_folds, list):
                index_to_skip = [i for i in self.skip_folds if i < len(folds)]        
        
        if self.verbose:
            self._print_info(
                index              = index,
                folds              = folds,
                externally_fitted  = externally_fitted,
                last_fold_excluded = last_fold_excluded,
                index_to_skip      = index_to_skip
            )

        folds = [fold for i, fold in enumerate(folds) if i not in index_to_skip]
        if not self.return_all_indexes:
            # +1 to prevent iloc pandas from deleting the last observation
            folds = [
                [[fold[0][0], fold[0][-1] + 1], 
                 [fold[1][0], fold[1][-1] + 1] if self.window_size is not None else [],
                 [fold[2][0], fold[2][-1] + 1],
                 [fold[3][0], fold[3][-1] + 1],
                 fold[4]] 
                for fold in folds
            ]

        if externally_fitted:
            self.initial_train_size = None
            folds[0][4] = False

        if as_pandas:
            if self.window_size is None:
                for fold in folds:
                    fold[1] = [None, None]

            if not self.return_all_indexes:
                folds = pd.DataFrame(
                    data = [list(itertools.chain(*fold[:-1])) + [fold[-1]] for fold in folds],
                    columns = [
                        'train_start',
                        'train_end',
                        'last_window_start',
                        'last_window_end',
                        'test_start',
                        'test_end',
                        'test_start_with_gap',
                        'test_end_with_gap',
                        'fit_forecaster'
                    ],
                )
            else:
                folds = pd.DataFrame(
                    data = folds,
                    columns = [
                        'train_index',
                        'last_window_index',
                        'test_index',
                        'test_index_with_gap',
                        'fit_forecaster'
                    ],
                )
            folds.insert(0, 'fold', range(len(folds)))

        return folds

    def _print_info(
        self,
        index: pd.Index,
        folds: list,
        externally_fitted: bool,
        last_fold_excluded: bool,
        index_to_skip: list,
    ) -> None:
        """
        Print information about folds.
        """

        print("Information of folds")
        print("--------------------")
        if externally_fitted:
            print(
                f"An already trained forecaster is to be used. Window size: "
                f"{self.window_size}"
            )
        elif self.initial_train_size == 0:
            print(
                f"Initial training size is 0. Window size: {self.window_size}"
            )
        else:
            if self.differentiation is None:
                print(
                    f"Number of observations used for initial training: "
                    f"{self.initial_train_size}"
                )
            else:
                print(
                    f"Number of observations used for initial training: "
                    f"{self.initial_train_size - self.differentiation}"
                )
                print(f"    First {self.differentiation} observation/s in training sets "
                        f"are used for differentiation"
                )
        print(
            f"Number of observations used for backtesting: "
            f"{len(index) - self.initial_train_size}"
        )
        print(f"    Number of folds: {len(folds)}")
        print(
            f"    Number skipped folds: "
            f"{len(index_to_skip)} {index_to_skip if index_to_skip else ''}"
        )
        print(f"    Number of steps per fold: {self.steps}")
        print(
            f"    Number of steps to exclude from the end of each train set "
            f"before test (gap): {self.gap}"
        )
        if last_fold_excluded:
            print("    Last fold has been excluded because it was incomplete.")
        if len(folds[-1][3]) < self.steps:
            print(f"    Last fold only includes {len(folds[-1][3])} observations.")
        print("")

        if self.differentiation is None:
            differentiation = 0
        else:
            differentiation = self.differentiation
        
        for i, fold in enumerate(folds):
            is_fold_skipped   = i in index_to_skip
            has_training      = fold[-1] if i != 0 else True
            training_start    = (
                index[fold[0][0] + differentiation] if fold[0] is not None else None
            )
            training_end      = index[fold[0][-1]] if fold[0] is not None else None
            training_length   = (
                len(fold[0]) - differentiation if fold[0] is not None else 0
            )
            validation_start  = index[fold[3][0]]
            validation_end    = index[fold[3][-1]]
            validation_length = len(fold[3])

            print(f"Fold: {i}")
            if is_fold_skipped:
                print("    Fold skipped")
            elif not externally_fitted and has_training:
                print(
                    f"    Training:   {training_start} -- {training_end}  "
                    f"(n={training_length})"
                )
                print(
                    f"    Validation: {validation_start} -- {validation_end}  "
                    f"(n={validation_length})"
                )
            else:
                print("    Training:   No training in this fold")
                print(
                    f"    Validation: {validation_start} -- {validation_end}  "
                    f"(n={validation_length})"
                )

        print("")
