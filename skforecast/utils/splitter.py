from __future__ import annotations
import pandas as pd
import numpy as np
import uuid
import sys
from typing import Literal
from .. import __version__
from .utils import check_preprocess_series
from ..preprocessing import reshape_series_wide_to_long


class TimeSeriesSplitter:
    """
    A utility class for splitting time series data into training, validation,
    and testing sets for machine learning algorithms.

    This class provides flexible splitting strategies supporting multiple input
    formats (wide DataFrame, long DataFrame with MultiIndex, or dictionary of Series),
    both DatetimeIndex and RangeIndex, and flexible output formats.

    **New in this version**: Support for multiple series arguments with independent
    splitting behavior. Each series can have different lengths and date ranges.

    Parameters
    ----------
    *series : pd.DataFrame | dict[str, pd.Series | pd.DataFrame]
        One or more time series data to split. Each can be:
        - Wide format pandas DataFrame with DatetimeIndex or RangeIndex
        - Long format pandas DataFrame with MultiIndex (series_id, datetime)
        - Dictionary of pandas Series or DataFrames with identical indexes

        When multiple series are provided, they are treated independently and
        splits are returned as a list of tuples (one tuple per series group).

    Attributes
    ----------
    series_groups_ : list[dict[str, pd.Series]]
        List of series dictionaries, one per input argument.
    series_indexes_ : list[dict[str, pd.Index]]
        List of index dictionaries, one per series group.
    n_groups_ : int
        Number of series groups (number of *series arguments).
    index_types_ : list[type]
        Type of index for each group (pd.DatetimeIndex or pd.RangeIndex).
    index_freqs_ : list[str | int | None]
        Frequency (for DatetimeIndex) or step (for RangeIndex) for each group.
    skforecast_version : str
        Version of skforecast library used to create the splitter.
    python_version : str
        Version of Python used to create the splitter.

    Raises
    ------
    ValueError
        If no series provided or series have invalid format.
    TypeError
        If inputs are not in supported format.

    Examples
    --------
    >>> import pandas as pd
    >>> from skforecast.utils.splitter import TimeSeriesSplitter

    >>> # Single series (backward compatible)
    >>> df1 = pd.DataFrame(
    ...     {'series_a': range(100), 'series_b': range(100, 200)},
    ...     index=pd.date_range('2023-01-01', periods=100, freq='d')
    ... )
    >>> splitter = TimeSeriesSplitter(df1)
    >>> train_set, test_set = splitter.split_by_date(
    ...     end_train='2023-03-11',
    ...     output_format='wide'
    ... )
    """

    def __init__(
        self, *series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame]
    ) -> None:
        """
        Initialize TimeSeriesSplitter with one or more series.

        Parameters
        ----------
        *series : pd.DataFrame | dict[str, pd.Series | pd.DataFrame]
            One or more time series data in supported formats.

        Raises
        ------
        ValueError
            If no series provided or series have invalid format.
        TypeError
            If series are not in a supported format.
        """
        if len(series) == 0:
            raise ValueError('At least one series must be provided.')

        # -- Process each series argument independently
        self.series_groups_ = []
        self.series_indexes_ = []
        self.index_types_ = []
        self.index_freqs_ = []
        self._min_indexes_ = []
        self._max_indexes_ = []

        for i, series_input in enumerate(series):
            # Use inner check_preprocess_series() preprocessing for each group
            series_dict, series_indexes = check_preprocess_series(series_input)

            # -- Store the preprocess series data dict & index dict
            self.series_groups_.append(series_dict)
            self.series_indexes_.append(series_indexes)

            # -- Store index type and frequency information for this group
            first_index = next(iter(series_indexes.values()))
            index_type = type(first_index)
            self.index_types_.append(index_type)

            if isinstance(first_index, pd.DatetimeIndex):
                self.index_freqs_.append(first_index.freq)
                self._min_indexes_.append(
                    min([idx.min() for idx in series_indexes.values()])
                )
                self._max_indexes_.append(
                    max([idx.max() for idx in series_indexes.values()])
                )
            if isinstance(first_index, pd.RangeIndex):
                self.index_freqs_.append(first_index.step)
                self._min_indexes_.append(0)
                self._max_indexes_.append(len(first_index) - 1)

        # -- Store the number groups/series input
        self.n_groups_ = len(series)
        self.n_timeseries = sum(map(len, self.series_indexes_))

        # -- Store version information
        self.skforecast_version = __version__
        self.python_version = sys.version.split(' ')[0]

    def _repr_html_(self) -> str:
        """
        Return HTML representation for Jupyter notebooks.

        Returns
        -------
        str
            HTML string with embedded CSS styling and object information.
        """
        # -- Define a component id
        unique_id = str(uuid.uuid4()).replace('-', '')

        # -- Define render colors
        background_color = '#f0f8ff'
        section_color = '#b3dbfd'

        # -- Define CSS styles
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
                max-width: 700px;
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
            .container-{unique_id} .group-section {{
                margin-left: 20px;
                padding: 5px;
                background-color: #ffffff;
                border-radius: 3px;
                margin-top: 5px;
            }}
        </style>
        """

        # -- Build series groups html content
        groups_html = ''
        for i in range(self.n_groups_):
            index_freq_info = (
                f'<strong>Frequency:</strong> {self.index_freqs_[i].freqstr}'
                if issubclass(self.index_types_[i], pd.DatetimeIndex)
                else f'<strong>Step:</strong> {self.index_freqs_[i]}'
            )
            n_timeseries = len(self.series_groups_[i])

            groups_html += f"""
                <div class="group-section">
                    <strong>Group {i}:</strong>
                    <ul style="margin-top: 3px;margin-bottom: 3px;">
                        <li><strong>Series count:</strong> {n_timeseries}</li>
                        <li><strong>Index type:</strong> {self.index_types_[i].__name__}</li>
                        <li>{index_freq_info}</li>
                        <li><strong>Range:</strong> {self._min_indexes_[i]} → {self._max_indexes_[i]}</li>
                    </ul>
                </div>
            """

        # -- Build global html content
        content = f"""
        <div class="container-{unique_id}">
            <h2>TimeSeriesSplitter</h2>
            <details open>
                <summary>Configuration</summary>
                <ul>
                    <li><strong>Number of groups:</strong> {self.n_groups_}</li>
                    <li><strong>Number of timeseries:</strong> {self.n_timeseries}</li>
                    <li><strong>Supported output formats:</strong> wide, long_multi_index, long, dict</li>
                </ul>
                {groups_html}
            </details>
            <details>
                <summary>Version Information</summary>
                <ul>
                    <li><strong>Skforecast version:</strong> {self.skforecast_version}</li>
                    <li><strong>Python version:</strong> {self.python_version}</li>
                </ul>
            </details>
        </div>
        """

        return style + content

    def _convert_date_to_position(
        self,
        date: str | pd.Timestamp,
        index: pd.Index,
        date_name: str = 'date',
    ) -> int:
        """
        Convert a date to its position in the given index.

        Parameters
        ----------
        date : str | pd.Timestamp
            Date to convert.
        index : pd.Index
            Index to search in.
        date_name : str, default 'date'
            Name of the date parameter (for error messages).

        Returns
        -------
        int
            Position of the date in the index.

        Raises
        ------
        ValueError
            If date is outside the valid range.
        """
        # -- Convert any string date input to Timestamp object
        if isinstance(date, str):
            date = pd.Timestamp(date)

        # -- Raise error if data is not in required time range
        if date not in index:
            raise ValueError(
                f'{date_name} {date} is not present in the series index. '
                f'Available range: {index[0]} to {index[-1]}.'
            )

        # -- Extract the numeric position
        return index.get_loc(date)

    def _validate_date_split_args(
        self,
        group_idx: int,
        start_train: str | pd.Timestamp | None,
        end_train: str | pd.Timestamp,
        end_validation: str | pd.Timestamp | None,
        end_test: str | pd.Timestamp | None,
    ) -> tuple[int, int, int | None, int | None]:
        """
        Validate and convert date split arguments to positions for a specific group.

        Parameters
        ----------
        group_idx : int
            Index of the series group to validate.

        Returns
        -------
        tuple[int, int, int | None, int | None]
            Positions for (start_train, end_train, end_validation, end_test).
        """
        # -- Force index to be DatetimeIndex object
        if self.index_types_[group_idx] != pd.DatetimeIndex:
            raise TypeError(
                f'Group {group_idx}: `split_by_date` requires `DatetimeIndex` object. '
                f'Current index type: {self.index_types_[group_idx].__name__}. '
                'Consider using `split_by_size` instead.'
            )

        first_index = next(iter(self.series_indexes_[group_idx].values()))

        # -- Convert dates to positions
        if start_train is None:
            start_train_pos = 0
        else:
            start_train_pos = self._convert_date_to_position(
                start_train, first_index, 'start_train'
            )

        end_train_pos = self._convert_date_to_position(
            end_train, first_index, 'end_train'
        )

        if end_validation is None:
            end_validation_pos = end_train_pos
        else:
            end_validation_pos = self._convert_date_to_position(
                end_validation, first_index, 'end_validation'
            )

        if end_test is None:
            end_test_pos = len(first_index) - 1
        else:
            end_test_pos = self._convert_date_to_position(
                end_test, first_index, 'end_test'
            )

        # -- Validate position order
        if start_train_pos >= end_train_pos:
            raise ValueError(
                f'Group {group_idx}: start_train must be earlier than end_train. '
                f'Got start_train={first_index[start_train_pos]}, '
                f'end_train={first_index[end_train_pos]}.'
            )

        if end_train_pos > end_validation_pos:
            raise ValueError(
                f'Group {group_idx}: end_train must be earlier than or equal to end_validation. '
                f'Got end_train={first_index[end_train_pos]}, '
                f'end_validation={first_index[end_validation_pos]}.'
            )

        if end_validation_pos > end_test_pos:
            raise ValueError(
                f'Group {group_idx}: end_validation must be earlier than or equal to end_test. '
                f'Got end_validation={first_index[end_validation_pos]}, '
                f'end_test={first_index[end_test_pos]}.'
            )

        return start_train_pos, end_train_pos, end_validation_pos, end_test_pos

    def _convert_size(
        self,
        size: int | float | None,
        size_name: str,
        total_len: int,
        group_idx: int | None = None,
    ) -> int | None:
        """
        Convert a size specification to an absolute integer count.

        This method handles both absolute (integer) and proportional (float)
        size specifications, validating the input and converting proportions
        to actual counts based on the total length.

        Parameters
        ----------
        size : int | float | None
            Size specification to convert:
            - If int: Absolute count (returned as-is after validation)
            - If float: Proportion of total_len (must be between 0 and 1)
            - If None: Returns None (indicates no size specified)
        size_name : str
            Name of the size parameter (e.g., 'train_size', 'validation_size').
            Used in error messages for clarity.
        total_len : int
            Total length of the series against which proportions are calculated.
        group_idx : int | None, default None
            Index of the series group being processed. If provided, it's included
            in error messages for multi-group scenarios.

        Returns
        -------
        int | None
            Absolute count as integer, or None if size was None.
            For float inputs, uses ceiling to ensure at least the requested
            proportion is included.
        """
        if size is None:
            return None

        # -- Build error message prefix with optional group index
        error_prefix = f'Group {group_idx}: ' if group_idx is not None else ''

        if isinstance(size, float):
            # -- Validate proportion is in valid range
            if not 0 < size < 1:
                raise ValueError(
                    f'{error_prefix}{size_name} proportion must be between 0 and 1. '
                    f'Got {size}.'
                )
            # -- Convert proportion to count using ceiling to ensure minimum coverage
            return int(np.ceil(size * total_len))

        # -- Return integer size as-is
        return int(size)

    def _validate_size_split_args(
        self,
        group_idx: int,
        train_size: int | float,
        validation_size: int | float | None,
        test_size: int | float | None,
    ) -> tuple[int, int | None, int | None]:
        """
        Validate and convert size split arguments for a specific group.

        Parameters
        ----------
        group_idx : int
            Index of the series group to validate.

        Returns
        -------
        tuple[int, int | None, int | None]
            Absolute sizes for (train, validation, test).
        """
        # -- Extract sample index (first one)
        first_index = next(iter(self.series_indexes_[group_idx].values()))
        total_len = len(first_index)

        # Convert all sizes using the helper method
        train_count = self._convert_size(train_size, 'train_size', total_len, group_idx)
        validation_count = self._convert_size(
            validation_size, 'validation_size', total_len, group_idx
        )
        test_count = self._convert_size(test_size, 'test_size', total_len, group_idx)

        # -- Validate total doesn't exceed series length
        total_requested = train_count
        if validation_count is not None:
            total_requested += validation_count
        if test_count is not None:
            total_requested += test_count

        if total_requested > total_len:
            raise ValueError(
                f'Group {group_idx}: Sum of requested sizes ({total_requested}) '
                f'exceeds series length ({total_len}). '
                f'Got train_size={train_count}, validation_size={validation_count}, '
                f'test_size={test_count}.'
            )

        # -- Return set sample counts
        return train_count, validation_count, test_count

    def _split_series_dict(
        self,
        series_dict: dict[str, pd.Series],
        positions: dict[str, tuple[int, int]],
    ) -> list[dict[str, pd.Series]]:
        """
        Split a single series dictionary according to positions.

        Parameters
        ----------
        series_dict : dict[str, pd.Series]
            Dictionary of series to split.
        positions : dict[str, tuple[int, int]]
            Start and end positions for each split.

        Returns
        -------
        list[dict[str, pd.Series]]
            List of dictionaries, one for each split.
        """
        # -- Collect series ids
        split_names = list(positions.keys())
        split_data = {name: {} for name in split_names}

        for series_name, series in series_dict.items():
            for split_name in split_names:
                start, end = positions[split_name]
                split_data[split_name][series_name] = series.iloc[
                    start : end + 1
                ].copy()

        return [split_data[name] for name in split_names]

    def _convert_output(
        self,
        split_dicts: list[dict[str, pd.Series]],
        output_format: Literal['wide', 'long', 'long_multi_index', 'dict'] = 'wide',
    ) -> tuple:
        """
        Convert split data to requested output format.

        Parameters
        ----------
        split_dicts : list[dict[str, pd.Series]]
            List of split data as dictionaries.
        output_format : {'wide', 'long', 'long_multi_index', 'dict'}, default 'wide'
            Output format.

        Returns
        -------
        tuple
            Splits in requested format.
        """
        match output_format:
            case 'dict':
                return tuple(split_dicts)

            case 'wide':
                return tuple(
                    pd.DataFrame.from_dict(split_dict) for split_dict in split_dicts
                )

            case 'long' | 'long_multi_index':
                return tuple(
                    reshape_series_wide_to_long(
                        pd.DataFrame.from_dict(split_dict),
                        return_multi_index=(output_format == 'long_multi_index'),
                    )
                    for split_dict in split_dicts
                )

            case _:
                raise ValueError(
                    f'Output format `{output_format}` is not supported. '
                    f'Choose one of ["wide", "long", "long_multi_index", "dict"].'
                )

    def split_by_date(
        self,
        end_train: str | pd.Timestamp,
        start_train: str | pd.Timestamp | None = None,
        end_validation: str | pd.Timestamp | None = None,
        end_test: str | pd.Timestamp | None = None,
        output_format: Literal['wide', 'long', 'long_multi_index', 'dict'] = 'wide',
        verbose: bool = False,
    ) -> list[tuple] | tuple:
        """
        Split time series based on date ranges.

        Creates training, validation (optional), and test sets by splitting
        series at specified date boundaries. Dates are inclusive.

        When multiple series groups were provided to the constructor, this method
        returns a list of tuples (one per group). Each group is split independently
        based on its own date range.

        Parameters
        ----------
        end_train : str | pd.Timestamp
            Training set end date (inclusive). Required parameter.
        start_train : str | pd.Timestamp | None, default None
            Training set start date (inclusive). Defaults to first date in each group.
        end_validation : str | pd.Timestamp | None, default None
            Validation set end date (inclusive).
            Defaults to end_train if not provided (no validation set created).
        end_test : str | pd.Timestamp | None, default None
            Test set end date (inclusive).
            Defaults to last date in each group.
        output_format : {'wide', 'long', 'long_multi_index', 'dict'}, default 'wide'
            Output format for the splits.
        verbose : bool, default False
            If True, print detailed split information for each group.

        Returns
        -------
        list[tuple] | tuple
            If single series group: tuple of splits (train, test) or (train, val, test)
            If multiple series groups: list of tuples, one per group

        Raises
        ------
        TypeError
            If series don't have DatetimeIndex.
        ValueError
            If dates are invalid or outside available range.

        Examples
        --------
        >>> # Single group
        >>> splitter = TimeSeriesSplitter(df1)
        >>> train, test = splitter.split_by_date(end_train='2023-03-11')

        >>> # Multiple groups
        >>> splitter = TimeSeriesSplitter(df1, df2, df3)
        >>> splits = splitter.split_by_date(end_train='2023-03-11')
        >>> # splits = [(df1_train, df1_test), (df2_train, df2_test), (df3_train, df3_test)]
        """
        results = []

        for group_idx in range(self.n_groups_):
            # -- Validate and get positions for current group
            start_pos, end_train_pos, end_val_pos, end_test_pos = (
                self._validate_date_split_args(
                    group_idx, start_train, end_train, end_validation, end_test
                )
            )

            # -- Define positions split dict
            positions = {'train': (start_pos, end_train_pos)}

            if end_validation is not None:
                positions['validation'] = (end_train_pos + 1, end_val_pos)
                positions['test'] = (end_val_pos + 1, end_test_pos)
            else:
                positions['test'] = (end_train_pos + 1, end_test_pos)

            # -- Perform split on current group
            split_dicts = [
                {k: v for k, v in split_dict.items() if len(v) > 0}
                for split_dict in self._split_series_dict(
                    self.series_groups_[group_idx], positions
                )
            ]

            # -- Convert to required output
            result = self._convert_output(split_dicts, output_format)

            if verbose:
                self._print_split_info(group_idx, positions, output_format)

            results.append(result)

        # -- Return single tuple if only one group, otherwise list of tuples
        return results if self.n_groups_ > 1 else results[0]

    def split_by_size(
        self,
        train_size: int | float,
        validation_size: int | float | None = None,
        test_size: int | float | None = None,
        output_format: Literal['wide', 'long', 'long_multi_index', 'dict'] = 'wide',
        verbose: bool = False,
    ) -> list[tuple] | tuple:
        """
        Split time series based on size (absolute or proportional).

        Creates training, validation (optional), and test sets by splitting
        series at specified size boundaries. Sizes can be absolute (int) or
        proportional (float between 0 and 1).

        When multiple series groups were provided to the constructor, this method
        returns a list of tuples (one per group). Each group is split independently
        based on its own length.

        Parameters
        ----------
        train_size : int | float
            Training set size. If int, absolute count. If float, proportion of total.
        validation_size : int | float | None, default None
            Validation set size. Same as train_size.
            If None, no validation set is created.
        test_size : int | float | None, default None
            Test set size. Same as train_size.
            If None, remainder is used as test set.
        output_format : {'wide', 'long', 'long_multi_index', 'dict'}, default 'wide'
            Output format for the splits.
        verbose : bool, default False
            If True, print detailed split information for each group.

        Returns
        -------
        list[tuple] | tuple
            If single series group: tuple of splits (train, test) or (train, val, test)
            If multiple series groups: list of tuples, one per group

        Raises
        ------
        ValueError
            If sizes are invalid or exceed series length.

        Examples
        --------
        >>> # Single group with proportions
        >>> splitter = TimeSeriesSplitter(df1)
        >>> train, test = splitter.split_by_size(train_size=0.8)

        >>> # Multiple groups with absolute sizes
        >>> splitter = TimeSeriesSplitter(df1, df2, df3)
        >>> splits = splitter.split_by_size(train_size=70, test_size=30)
        >>> # Each group split with 70 training samples and 30 test samples
        """
        results = []

        for group_idx in range(self.n_groups_):
            # -- Validate and get counts for current group
            train_count, val_count, test_count = self._validate_size_split_args(
                group_idx, train_size, validation_size, test_size
            )

            # -- Get total length for current group
            first_index = next(iter(self.series_indexes_[group_idx].values()))
            total_len = len(first_index)

            # -- Compute positions
            train_end = train_count - 1
            val_end = train_end + (val_count if val_count is not None else 0)
            test_end = total_len - 1

            positions = {'train': (0, train_end)}

            if val_count is not None:
                positions['validation'] = (train_end + 1, val_end)
                positions['test'] = (val_end + 1, test_end)
            else:
                positions['test'] = (train_end + 1, test_end)

            # -- Perform split on current group
            split_dicts = self._split_series_dict(
                self.series_groups_[group_idx], positions
            )

            # -- Convert to required output
            result = self._convert_output(split_dicts, output_format)

            if verbose:
                self._print_split_info(group_idx, positions, output_format)

            results.append(result)

        # Return single tuple if only one group, otherwise list of tuples
        return results[0] if self.n_groups_ == 1 else results

    def _print_split_info(
        self,
        group_idx: int,
        positions: dict[str, tuple[int, int]],
        output_format: str,
    ) -> None:
        """
        Print detailed split information for a specific group.

        Parameters
        ----------
        group_idx : int
            Index of the series group.
        positions : dict[str, tuple[int, int]]
            Position ranges for each split.
        output_format : str
            Output format being used.
        """
        # -- Extract a sample index (first one)
        first_index = next(iter(self.series_indexes_[group_idx].values()))
        total_len = len(first_index)

        # -- Print header
        print(f'Split Information (Group id: {group_idx})')
        print('=' * 32)

        for split_name, (start, end) in positions.items():
            length = max(0, end - start + 1)
            percentage = (length / total_len * 100) if total_len > 0 else 0

            if isinstance(first_index, pd.DatetimeIndex):
                start_date = first_index[start] if start < len(first_index) else 'N/A'
                end_date = first_index[end] if end < len(first_index) else 'N/A'
                print(
                    f'{split_name.capitalize():12} | '
                    f'Range: {start_date} to {end_date} | '
                    f'Length: {length} ({percentage:.1f}%)'
                )
            else:
                print(
                    f'{split_name.capitalize():12} | '
                    f'Positions: {start} to {end} | '
                    f'Length: {length} ({percentage:.1f}%)'
                )
        # -- Print the given output format
        print(f'Output format: {output_format}', end='')
        if group_idx < self.n_groups_:
            print('\n')
