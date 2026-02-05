from __future__ import annotations
import pandas as pd
import numpy as np
import uuid
import sys
from .. import __version__


class TimeSeriesSplitter:
    """
    A utility class for splitting time series data into training, validation,
    and testing sets for machine learning algorithms.

    This class provides flexible splitting strategies for one or multiple
    DataFrames with support for both date-based and size-based splitting methods.
    Only supports wide-format DataFrames with DatetimeIndex.

    Parameters
    ----------
    *args : pd.DataFrame
        One or more pandas DataFrames to be split. DataFrames must have:
        - A DatetimeIndex (wide format)
        - A sorted datetime index in ascending order

    Attributes
    ----------
    dataframes_ : list[pd.DataFrame]
        List of copied DataFrames as provided to the constructor.
    n_dataframes_ : int
        Number of DataFrames stored.
    _min_date : pd.Timestamp
        Minimum date across all stored DataFrames.
    _max_date : pd.Timestamp
        Maximum date across all stored DataFrames.
    skforecast_version : str
        Version of skforecast library used to create the splitter.
    python_version : str
        Version of Python used to create the splitter.

    Raises
    ------
    ValueError
        If no DataFrames are provided, if DataFrames don't have DatetimeIndex,
        or if DatetimeIndex is not sorted in ascending order.
    TypeError
        If inputs are not pandas DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> from skforecast.utils import TimeSeriesSplitter

    >>> # Wide format (single DataFrame with DatetimeIndex)
    >>> df = pd.DataFrame(
    ...     {'value': range(100)},
    ...     index=pd.date_range('2023-01-01', periods=100, freq='D')
    ... )
    >>> splitter = TimeSeriesSplitter(df)
    >>> train, test = splitter.split_by_date(
    ...     end_train='2023-03-11',
    ...     end_test='2023-04-10',
    ...     verbose=True
    ... )

    >>> # Multiple DataFrames
    >>> df1 = pd.DataFrame(...)  # DatetimeIndex
    >>> df2 = pd.DataFrame(...)  # DatetimeIndex
    >>> splitter = TimeSeriesSplitter(df1, df2)
    >>> splits = splitter.split_by_date(end_train='2023-03-11', verbose=True)

    >>> # Size-based splitting
    >>> train, val, test = splitter.split_by_size(
    ...     train_size=60,
    ...     validation_size=20,
    ...     test_size=20,
    ...     verbose=True
    ... )
    """

    def __init__(self, *args: pd.DataFrame) -> None:
        """
        Initialize TimeSeriesSplitter with validation.

        Validates that all input DataFrames have DatetimeIndex and are sorted,
        then stores them for later splitting operations.

        Parameters
        ----------
        *args : pd.DataFrame
            One or more pandas DataFrames to store and split.

        Raises
        ------
        ValueError
            If no DataFrames provided, DataFrame lacks DatetimeIndex, or
            DatetimeIndex is not sorted in ascending order.
        TypeError
            If any argument is not a pandas DataFrame.
        """

        if len(args) == 0:
            raise ValueError('At least one DataFrame must be provided.')

        # Type checking
        for i, arg in enumerate(args):
            if not isinstance(arg, pd.DataFrame):
                raise TypeError(
                    f'Argument {i} must be a pandas DataFrame. '
                    f'Got {type(arg).__name__}.'
                )

        # Validate all DataFrames have DatetimeIndex and sorted
        for i, df in enumerate(args):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    f'DataFrame {i} must have a DatetimeIndex. '
                    f'Got {type(df.index).__name__}. '
                    f'TimeSeriesSplitter only supports wide-format DataFrames with DatetimeIndex.'
                )

            # Check if index is sorted
            if not df.index.is_monotonic_increasing:
                raise ValueError(
                    f'DataFrame {i} has an unsorted DatetimeIndex. '
                    f'DatetimeIndex must be sorted in ascending order for time series analysis.'
                )

        # Store dataframes directly (copy to prevent external modifications)
        self.dataframes_ = [df.copy(deep=True) for df in args]
        self.n_dataframes_ = len(args)

        # Store min/max dates across all input DataFrames for reference
        self._min_date = min([df.index.min() for df in self.dataframes_])
        self._max_date = max([df.index.max() for df in self.dataframes_])

        # Store version information
        self.skforecast_version = __version__
        self.python_version = sys.version.split(' ')[0]

    def __repr__(self) -> str:
        """
        Return string representation of TimeSeriesSplitter.

        Returns
        -------
        str
            Formatted string showing class name, number of DataFrames,
            format type, and overall date range.
        """
        return (
            f'{"=" * 51}\n'
            f'{"TimeSeriesSplitter":<51}\n'
            f'{"=" * 51}\n'
            f'Number of DataFrames: {self.n_dataframes_}\n'
            f'Format: Wide format (DatetimeIndex)\n'
            f'Overall date range: {self._min_date} to {self._max_date}\n'
        )

    def _repr_html_(self) -> str:
        """
        Return HTML representation for Jupyter notebooks.

        Generates a styled HTML representation with configuration details,
        using a unique UUID to ensure CSS class name uniqueness.

        Returns
        -------
        str
            HTML string with embedded CSS styling and object information.
        """

        unique_id = str(uuid.uuid4()).replace('-', '')
        background_color = '#f0f8ff'
        section_color = '#b3dbfd'

        style = f"""
        <style>
            .container-{unique_id} {{
                font-family: 'Arial', sans-serif;
                font-size: 0.9em;
                color: #333333;
                border: 1px solid #ddd;
                background-color: {background_color};
                padding: 5px 15px;
                border-radius: 8px;
                max-width: 600px;
            }}
            .container-{unique_id} h2 {{
                font-size: 1.5em;
                color: #222222;
                border-bottom: 2px solid #ddd;
                padding-bottom: 5px;
                margin-bottom: 15px;
                margin-top: 5px;
            }}
            .container-{unique_id} details {{
                margin: 10px 0;
            }}
            .container-{unique_id} summary {{
                font-weight: bold;
                font-size: 1.1em;
                color: #000000;
                cursor: pointer;
                margin-bottom: 5px;
                background-color: {section_color};
                padding: 5px;
                border-radius: 5px;
            }}
            .container-{unique_id} summary:hover {{
                color: #000000;
                background-color: #e0e0e0;
            }}
            .container-{unique_id} ul {{
                font-family: 'Courier New', monospace;
                list-style-type: none;
                padding-left: 20px;
                margin: 10px 0;
                line-height: normal;
            }}
            .container-{unique_id} li {{
                margin: 5px 0;
                font-family: 'Courier New', monospace;
            }}
            .container-{unique_id} li strong {{
                font-weight: bold;
                color: #444444;
            }}
            .container-{unique_id} li::before {{
                content: "- ";
                color: #666666;
            }}
            .container-{unique_id} a {{
                color: #001633;
                text-decoration: none;
            }}
            .container-{unique_id} a:hover {{
                color: #359ccb; 
            }}
        </style>
        """

        content = f"""
        <div class="container-{unique_id}">
            <h2>TimeSeriesSplitter</h2>
            <details open>
                <summary>Configuration</summary>
                <ul>
                    <li><strong>Number of DataFrames:</strong> {self.n_dataframes_}</li>
                    <li><strong>Format:</strong> Wide format (DatetimeIndex)</li>
                    <li><strong>Overall date range:</strong> {self._min_date} → {self._max_date}</li>
                    <li><strong>Splitting Methods:</strong> split_by_date, split_by_size</li>
                    <li><strong>Skforecast version:</strong> {self.skforecast_version}</li>
                    <li><strong>Python version:</strong> {self.python_version}</li>
                </ul>
            </details>
            <p style="margin-top: 15px; font-size: 0.9em;">
                <strong>Usage:</strong> Use <code>split_by_date()</code> or <code>split_by_size()</code> 
                methods with <code>verbose=True</code> to see detailed split.
            </p>
        </div>
        """

        return style + content

    def split_by_date(
        self,
        end_train: str | pd.Timestamp,
        start_train: str | pd.Timestamp | None = None,
        end_validation: str | pd.Timestamp | None = None,
        end_test: str | pd.Timestamp | None = None,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, ...] | list[tuple[pd.DataFrame, ...]]:
        """
        Split time series DataFrames based on date ranges.

        Creates training, validation (optional), and test sets by splitting
        DataFrames at specified date boundaries. Dates are inclusive.

        Parameters
        ----------
        end_train : str | pd.Timestamp
            Training set end date (inclusive). Required parameter.
        start_train : str | pd.Timestamp | None, optional
            Training set start date (inclusive). Defaults to minimum date in data.
        end_validation : str | pd.Timestamp | None, optional
            Validation set end date (inclusive).
            Defaults to end_train if not provided (no validation set created).
        end_test : str | pd.Timestamp | None, optional
            Test set end date (inclusive).
            Defaults to maximum date in data.
        verbose : bool, default False
            If True, print detailed split information including date ranges,
            counts, and percentages for each set.

        Returns
        -------
        tuple[pd.DataFrame, ...] | list[tuple[pd.DataFrame, ...]]
            If single DataFrame was provided:
                Tuple of (train, test) or (train, validation, test) DataFrames.
            If multiple DataFrames were provided:
                List of tuples, one per input DataFrame.

        Raises
        ------
        ValueError
            If dates are invalid or outside available range.

        Notes
        -----
        - Validation set starts the day after end_train
        - Test set starts the day after end_validation
        - Dates are inclusive on both boundaries
        - Date boundaries must satisfy: start_train < end_train <= end_validation < end_test

        Examples
        --------
        >>> splitter = TimeSeriesSplitter(df)
        >>> train, test = splitter.split_by_date(
        ...     end_train='2023-03-11',
        ...     end_test='2023-04-10'
        ... )

        >>> # With validation set
        >>> train, val, test = splitter.split_by_date(
        ...     end_train='2023-03-11',
        ...     end_validation='2023-03-25',
        ...     end_test='2023-04-10',
        ...     verbose=True
        ... )
        """
        splits = []
        verbose_messages = []

        # Iterate through stored dataframes
        for i, df in enumerate(self.dataframes_):
            # Get date range
            df_min, df_max = df.index.min(), df.index.max()

            # Convert date strings to Timestamps
            _start_train = df_min if start_train is None else pd.Timestamp(start_train)
            _end_train = pd.Timestamp(end_train)
            _end_validation = (
                pd.Timestamp(end_validation)
                if end_validation is not None
                else _end_train
            )
            _end_test = df_max if end_test is None else pd.Timestamp(end_test)

            # Validate date ordering
            if _start_train < df_min:
                raise ValueError(
                    f'`start_train` ({_start_train}) cannot be earlier than '
                    f'minimum index date ({df_min}).'
                )

            if _start_train >= _end_train:
                raise ValueError(
                    f'`start_train` ({_start_train}) must be earlier than '
                    f'`end_train` ({_end_train}).'
                )

            if _end_train > _end_validation:
                raise ValueError(
                    f'`end_train` ({_end_train}) must be earlier than or equal to '
                    f'`end_validation` ({_end_validation}).'
                )

            if _end_validation >= _end_test:
                raise ValueError(
                    f'`end_validation` ({_end_validation}) must be earlier than '
                    f'`end_test` ({_end_test}).'
                )

            if _end_test > df_max:
                raise ValueError(
                    f'`end_test` ({_end_test}) cannot be later than '
                    f'maximum index date ({df_max}).'
                )

            # Split based on date ranges
            df_train = df.loc[_start_train:_end_train]
            df_validation = (
                df.loc[_end_train:_end_validation].iloc[1:]
                if _end_validation > _end_train
                else pd.DataFrame(columns=df.columns)
            )
            df_test = (
                df.loc[_end_validation:_end_test].iloc[1:]
                if _end_test > _end_validation
                else pd.DataFrame(columns=df.columns)
            )

            # Create split tuple
            if end_validation is None or (len(df_validation) == 0):
                split = (df_train, df_test)
            else:
                split = (df_train, df_validation, df_test)

            splits.append(split)

            # Collect verbose info
            if verbose:
                verbose_messages.append(self._get_split_info_message(i, split))

        # Print verbose info
        if verbose and verbose_messages:
            print('\n'.join(verbose_messages))

        # Return results
        if self.n_dataframes_ == 1:
            return splits[0]
        return splits

    def split_by_size(
        self,
        train_size: float | int = None,
        validation_size: float | int | None = None,
        test_size: float | int | None = None,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, ...] | list[tuple[pd.DataFrame, ...]]:
        """
        Split time series DataFrames based on size ratios or counts.

        Creates training, validation (optional), and test sets by splitting
        DataFrames into specified sizes. Sizes can be proportions (0-1) or
        absolute counts. Time series ordering is always preserved.

        Parameters
        ----------
        train_size : float | int
            Size of training set. Required parameter.
            If float (0-1), interpreted as proportion of total data.
            If int, interpreted as absolute number of samples.
        validation_size : float | int | None, optional
            Size of validation set.
            If not provided, no validation set is created.
            If float (0-1), interpreted as proportion of total data.
            If int, interpreted as absolute number of samples.
        test_size : float | int | None, optional
            Size of test set.
            If not provided, remaining data after train and validation is used.
            If float (0-1), interpreted as proportion of total data.
            If int, interpreted as absolute number of samples.
        verbose : bool, default False
            If True, print detailed split information including date ranges,
            counts, and percentages for each set.

        Returns
        -------
        tuple[pd.DataFrame, ...] | list[tuple[pd.DataFrame, ...]]
            If single DataFrame was provided:
                Tuple of (train, test) or (train, validation, test) DataFrames.
            If multiple DataFrames were provided:
                List of tuples, one per input DataFrame.

        Raises
        ------
        ValueError
            If:
            - train_size is not provided
            - Any float size is not in range [0, 1]
            - Any int size is negative
            - Total split size exceeds DataFrame length
        TypeError
            If size parameters are not float, int, or None.

        Notes
        -----
        - Float sizes are converted to int using np.ceil
        - Proportions are based on total DataFrame length
        - Time series ordering is always preserved (no shuffling)
        - Empty validation/test sets return empty DataFrames with correct columns

        Examples
        --------
        >>> splitter = TimeSeriesSplitter(df)

        >>> # Using proportions
        >>> train, test = splitter.split_by_size(
        ...     train_size=0.7,
        ...     test_size=0.3
        ... )

        >>> # Using absolute counts
        >>> train, val, test = splitter.split_by_size(
        ...     train_size=60,
        ...     validation_size=20,
        ...     test_size=20,
        ...     verbose=True
        ... )
        """
        splits = []
        verbose_messages = []

        for i, df in enumerate(self.dataframes_):
            # Get length
            n_samples = len(df)

            # Convert proportion sizes to absolute counts
            _train_size = self._convert_size(train_size, n_samples)
            _validation_size = (
                self._convert_size(validation_size, n_samples)
                if validation_size is not None
                else None
            )
            _test_size = (
                self._convert_size(test_size, n_samples)
                if test_size is not None
                else None
            )

            # Validate sizes
            total_size = _train_size + (_validation_size or 0) + (_test_size or 0)
            if total_size > n_samples:
                raise ValueError(
                    f'Total split size ({total_size}) exceeds DataFrame length ({n_samples}).'
                )

            # Split using indices (preserve time series ordering)
            idx_train_end = _train_size
            idx_val_end = idx_train_end + (_validation_size or 0)

            # Create splits
            df_train = df.iloc[:idx_train_end]
            df_validation = (
                df.iloc[idx_train_end:idx_val_end]
                if validation_size is not None
                else pd.DataFrame(columns=df.columns)
            )
            df_test = df.iloc[idx_val_end:]

            # Create split tuple
            if validation_size is None or len(df_validation) == 0:
                split = (df_train, df_test)
            else:
                split = (df_train, df_validation, df_test)

            splits.append(split)

            # Collect verbose info
            if verbose:
                verbose_messages.append(self._get_split_info_message(i, split))

        # Print verbose info
        if verbose and verbose_messages:
            print('\n'.join(verbose_messages))

        # Return results
        if self.n_dataframes_ == 1:
            return splits[0]
        return splits

    @staticmethod
    def _convert_size(size: float | int | None, total: int) -> int:
        """
        Convert size specification to absolute count.

        Converts a size specification (either a proportion as float or absolute
        count as int) to an absolute integer count of samples.

        Parameters
        ----------
        size : float | int | None
            Size specification. If float, must be between 0 and 1 (inclusive).
            If int, must be non-negative. If None, returns 0.
        total : int
            Total number of samples available. Used to compute absolute count
            when size is a proportion.

        Returns
        -------
        int
            Absolute count of samples. For float inputs, returns ceil(size * total).
            For int inputs, returns size directly. For None, returns 0.

        Raises
        ------
        ValueError
            If float size is not between 0 and 1 (inclusive),
            or if int size is negative.
        TypeError
            If size is neither float, int, nor None.

        Examples
        --------
        >>> TimeSeriesSplitter._convert_size(0.6, 100)
        60
        >>> TimeSeriesSplitter._convert_size(0.55, 100)
        55
        >>> TimeSeriesSplitter._convert_size(50, 100)
        50
        >>> TimeSeriesSplitter._convert_size(None, 100)
        0
        """
        if size is None:
            return 0
        if isinstance(size, float):
            if not 0 <= size <= 1:
                raise ValueError(f'Float size must be between 0 and 1. Got {size}.')
            return int(np.ceil(size * total))
        elif isinstance(size, int):
            if size < 0:
                raise ValueError(f'Integer size must be non-negative. Got {size}.')
            return size
        else:
            raise TypeError(f'Size must be float or int. Got {type(size).__name__}.')

    def _get_split_info_message(self, idf: int, split: tuple[pd.DataFrame, ...]) -> str:
        """
        Generate verbose information message for split.

        Creates a formatted string showing split information including date ranges,
        sizes, and percentages for train, validation, and test sets.

        Parameters
        ----------
        idf : int
            DataFrame index (0-based) for labeling in the output.
        split : tuple[pd.DataFrame, ...]
            Tuple of DataFrames representing the split. Can contain 2 or 3 DataFrames
            (train, test) or (train, validation, test).

        Returns
        -------
        str
            Formatted string with split information. Each line contains:
            - Set label (Train/Validation/Test)
            - Percentage of total data
            - Date range (min → max)
            - Number of samples [n=X]

        Notes
        -----
        - Empty DataFrames (zero length) are not displayed in the output
        - Percentages are calculated as (set_size / total_size) * 100
        - If no data is in any set, percentage displays as "N/A"
        """
        header = f'DataFrame {idf + 1} - Split Information:'
        n_total = sum(len(s) for s in split)
        lines = [header]
        labels = ['Train', 'Validation', 'Test'] if len(split) == 3 else ['Train', 'Test']
        for label, df in zip(labels, split):
            if len(df) > 0:
                percentage = f'{len(df) / n_total:.1%}' if n_total > 0 else 'N/A'
                date_range = f'{df.index.min()} → {df.index.max()}'
                lines.append(
                    f'  {label:12} ({percentage:>5}): {date_range:30} [n={len(df)}]'
                )

        return '\n'.join(lines)
