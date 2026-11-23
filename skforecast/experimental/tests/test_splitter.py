# Unit test TimeSeriesSplitter class & methods
# ==============================================================================
import sys
import pytest
import numpy as np
import pandas as pd

from .._splitter import TimeSeriesSplitter

pytestmark = [
    pytest.mark.filterwarnings("ignore::skforecast.exceptions.InputTypeWarning"),
]


@pytest.fixture
def df_datetime_wide():
    """Wide format DataFrame with DatetimeIndex."""
    return pd.DataFrame(
        {
            'series_a': range(100),
            'series_b': range(100, 200),
            'series_c': range(200, 300),
        },
        index=pd.date_range('2023-01-01', periods=100, freq='D'),
    )


@pytest.fixture
def df_datetime_wide_short():
    """Short wide format DataFrame with DatetimeIndex."""
    return pd.DataFrame(
        {'series_a': range(10), 'series_b': range(10, 20)},
        index=pd.date_range('2023-01-01', periods=10, freq='D'),
    )


@pytest.fixture
def df_range_wide():
    """Wide format DataFrame with RangeIndex."""
    return pd.DataFrame(
        {'series_a': range(100), 'series_b': range(100, 200)},
        index=pd.RangeIndex(start=0, stop=100, step=1),
    )


@pytest.fixture
def df_datetime_long():
    """Long format DataFrame with MultiIndex."""
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    series_ids = ['series_a', 'series_b']

    index = pd.MultiIndex.from_product(
        [series_ids, dates], names=['series_id', 'datetime']
    )

    return pd.DataFrame({'value': range(100)}, index=index)


@pytest.fixture
def dict_series_datetime():
    """Dictionary of Series with DatetimeIndex."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    return {
        'series_a': pd.Series(range(100), index=dates),
        'series_b': pd.Series(range(100, 200), index=dates),
    }


@pytest.fixture
def dict_series_range():
    """Dictionary of Series with RangeIndex."""
    return {
        'series_a': pd.Series(range(100), index=pd.RangeIndex(0, 100, 1)),
        'series_b': pd.Series(range(100, 200), index=pd.RangeIndex(0, 100, 1)),
    }


@pytest.fixture
def multiple_dfs():
    """Multiple DataFrames for multi-group testing."""
    df1 = pd.DataFrame(
        {'s1': range(100)}, index=pd.date_range('2023-01-01', periods=100, freq='D')
    )
    df2 = pd.DataFrame(
        {'s2': range(50)}, index=pd.date_range('2023-01-01', periods=50, freq='D')
    )
    df3 = pd.DataFrame(
        {'s3': range(80)}, index=pd.date_range('2023-01-01', periods=80, freq='D')
    )
    return df1, df2, df3


def expected_long(series, return_multi_index, tz=None):
    """
    Build the expected long DataFrame from the dates and the values of each
    series: {series_id: (dates, values)}. With `tz`, the dates are written with
    their UTC offset.
    """
    series_ids = [k for k, (dates, _) in series.items() for _ in dates]
    dates = [date for dates, _ in series.values() for date in dates]
    values = [value for _, values in series.values() for value in values]
    if tz is None:
        dates = pd.DatetimeIndex(dates)
    else:
        dates = pd.to_datetime(dates, utc=True).tz_convert(tz)
    expected = pd.DataFrame(
        {'value': values},
        index=pd.MultiIndex.from_arrays(
            [series_ids, dates], names=['series_id', 'datetime']
        ),
    )

    return expected if return_multi_index else expected.reset_index()


# Expected long outputs of `split_by_size` with two series with the same
# frequency and number of values: `a` starts on 2023-01-01 and `b` on `start_b`.
# When the dates of both are in the same grid, they are filled with NaN up to
# the span of the split (with 'MS', `date_range` moves the start of `b`,
# 2023-01-15, to the next month start). Otherwise, each keeps its own dates.
nan = np.nan
long_cases = {
    '7D + 3D': {
        'freq': '7D',
        'start_b': '2023-01-04',
        'periods': 10,
        'train_size': 7,
        'train': {
            'a': (
                ['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22',
                 '2023-01-29', '2023-02-05', '2023-02-12'],
                [0., 1., 2., 3., 4., 5., 6.],
            ),
            'b': (
                ['2023-01-04', '2023-01-11', '2023-01-18', '2023-01-25',
                 '2023-02-01', '2023-02-08', '2023-02-15'],
                [100., 101., 102., 103., 104., 105., 106.],
            ),
        },
        'test': {
            'a': (['2023-02-19', '2023-02-26', '2023-03-05'], [7., 8., 9.]),
            'b': (['2023-02-22', '2023-03-01', '2023-03-08'], [107., 108., 109.]),
        },
    },
    'D + 6h': {
        'freq': 'D',
        'start_b': '2023-01-01 06:00',
        'periods': 10,
        'train_size': 7,
        'train': {
            'a': (
                ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
                 '2023-01-05', '2023-01-06', '2023-01-07'],
                [0., 1., 2., 3., 4., 5., 6.],
            ),
            'b': (
                ['2023-01-01 06:00', '2023-01-02 06:00', '2023-01-03 06:00',
                 '2023-01-04 06:00', '2023-01-05 06:00', '2023-01-06 06:00',
                 '2023-01-07 06:00'],
                [100., 101., 102., 103., 104., 105., 106.],
            ),
        },
        'test': {
            'a': (['2023-01-08', '2023-01-09', '2023-01-10'], [7., 8., 9.]),
            'b': (
                ['2023-01-08 06:00', '2023-01-09 06:00', '2023-01-10 06:00'],
                [107., 108., 109.],
            ),
        },
    },
    '3h + 1h': {
        'freq': '3h',
        'start_b': '2023-01-01 01:00',
        'periods': 10,
        'train_size': 7,
        'train': {
            'a': (
                ['2023-01-01 00:00', '2023-01-01 03:00', '2023-01-01 06:00',
                 '2023-01-01 09:00', '2023-01-01 12:00', '2023-01-01 15:00',
                 '2023-01-01 18:00'],
                [0., 1., 2., 3., 4., 5., 6.],
            ),
            'b': (
                ['2023-01-01 01:00', '2023-01-01 04:00', '2023-01-01 07:00',
                 '2023-01-01 10:00', '2023-01-01 13:00', '2023-01-01 16:00',
                 '2023-01-01 19:00'],
                [100., 101., 102., 103., 104., 105., 106.],
            ),
        },
        'test': {
            'a': (
                ['2023-01-01 21:00', '2023-01-02 00:00', '2023-01-02 03:00'],
                [7., 8., 9.],
            ),
            'b': (
                ['2023-01-01 22:00', '2023-01-02 01:00', '2023-01-02 04:00'],
                [107., 108., 109.],
            ),
        },
    },
    'MS + 14D': {
        'freq': 'MS',
        'start_b': '2023-01-15',
        'periods': 10,
        'train_size': 7,
        'train': {
            'a': (
                ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01',
                 '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01'],
                [0., 1., 2., 3., 4., 5., 6., nan],
            ),
            'b': (
                ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01',
                 '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01'],
                [nan, 100., 101., 102., 103., 104., 105., 106.],
            ),
        },
        'test': {
            'a': (
                ['2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01'],
                [7., 8., 9., nan],
            ),
            'b': (
                ['2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01'],
                [nan, 107., 108., 109.],
            ),
        },
    },
    # Same grid, spans separated by a gap: the union of the dates has no
    # frequency.
    'D, gap between spans': {
        'freq': 'D',
        'start_b': '2023-01-08',
        'periods': 6,
        'train_size': 4,
        'train': {
            'a': (
                ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
                 '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
                 '2023-01-09', '2023-01-10', '2023-01-11'],
                [0., 1., 2., 3., nan, nan, nan, nan, nan, nan, nan],
            ),
            'b': (
                ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
                 '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
                 '2023-01-09', '2023-01-10', '2023-01-11'],
                [nan, nan, nan, nan, nan, nan, nan, 100., 101., 102., 103.],
            ),
        },
        'test': {
            'a': (
                ['2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
                 '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12',
                 '2023-01-13'],
                [4., 5., nan, nan, nan, nan, nan, nan, nan],
            ),
            'b': (
                ['2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
                 '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12',
                 '2023-01-13'],
                [nan, nan, nan, nan, nan, nan, nan, 104., 105.],
            ),
        },
    },
    # Same grid, one date per series in the test split: with two dates, pandas
    # 2 gives their union a frequency of 3 days.
    'D, one date per series': {
        'freq': 'D',
        'start_b': '2023-01-04',
        'periods': 5,
        'train_size': 4,
        'train': {
            'a': (
                ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
                 '2023-01-05', '2023-01-06', '2023-01-07'],
                [0., 1., 2., 3., nan, nan, nan],
            ),
            'b': (
                ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
                 '2023-01-05', '2023-01-06', '2023-01-07'],
                [nan, nan, nan, 100., 101., 102., 103.],
            ),
        },
        'test': {
            'a': (
                ['2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'],
                [4., nan, nan, nan],
            ),
            'b': (
                ['2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'],
                [nan, nan, nan, 104.],
            ),
        },
    },
    # Shifted dates whose union is regular: pandas gives it a frequency of 12
    # hours.
    'D + 12h': {
        'freq': 'D',
        'start_b': '2023-01-01 12:00',
        'periods': 6,
        'train_size': 4,
        'train': {
            'a': (
                ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
                [0., 1., 2., 3.],
            ),
            'b': (
                ['2023-01-01 12:00', '2023-01-02 12:00', '2023-01-03 12:00',
                 '2023-01-04 12:00'],
                [100., 101., 102., 103.],
            ),
        },
        'test': {
            'a': (['2023-01-05', '2023-01-06'], [4., 5.]),
            'b': (['2023-01-05 12:00', '2023-01-06 12:00'], [104., 105.]),
        },
    },
    # Shifted dates whose union gets the frequency of the series: pandas gives
    # a monthly frequency to month starts at 00:00 followed by others at 06:00.
    'MS + 6h': {
        'freq': 'MS',
        'start_b': '2023-05-01 06:00',
        'periods': 6,
        'train_size': 4,
        'train': {
            'a': (
                ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'],
                [0., 1., 2., 3.],
            ),
            'b': (
                ['2023-05-01 06:00', '2023-06-01 06:00', '2023-07-01 06:00',
                 '2023-08-01 06:00'],
                [100., 101., 102., 103.],
            ),
        },
        'test': {
            'a': (['2023-05-01', '2023-06-01'], [4., 5.]),
            'b': (['2023-09-01 06:00', '2023-10-01 06:00'], [104., 105.]),
        },
    },
}


class TestInit:
    """Test the __init__ method of TimeSeriesSplitter."""

    def test_init_no_series_raises_error(self):
        """Test that initializing without series raises ValueError."""
        with pytest.raises(ValueError, match='At least one series must be provided'):
            TimeSeriesSplitter()

    def test_init_single_datetime_wide(self, df_datetime_wide):
        """Test initialization with single wide DataFrame with DatetimeIndex."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        assert splitter.n_groups_ == 1
        assert splitter.n_timeseries == 3
        assert len(splitter.series_groups_) == 1
        assert len(splitter.series_groups_[0]) == 3
        assert splitter.index_types_[0] == pd.DatetimeIndex
        assert splitter.python_version == sys.version.split(' ')[0]

    def test_init_single_range_wide(self, df_range_wide):
        """Test initialization with single wide DataFrame with RangeIndex."""
        splitter = TimeSeriesSplitter(df_range_wide)

        assert splitter.n_groups_ == 1
        assert splitter.n_timeseries == 2
        assert splitter.index_types_[0] == pd.RangeIndex
        assert splitter.index_freqs_[0] == 1
        assert splitter._min_indexes_[0] == 0
        assert splitter._max_indexes_[0] == 99

    def test_init_single_dict_datetime(self, dict_series_datetime):
        """Test initialization with dictionary of Series with DatetimeIndex."""
        splitter = TimeSeriesSplitter(dict_series_datetime)

        assert splitter.n_groups_ == 1
        assert splitter.n_timeseries == 2
        assert splitter.index_types_[0] == pd.DatetimeIndex
        assert len(splitter.series_groups_[0]) == 2

    def test_init_single_dict_range(self, dict_series_range):
        """Test initialization with dictionary of Series with RangeIndex."""
        splitter = TimeSeriesSplitter(dict_series_range)

        assert splitter.n_groups_ == 1
        assert splitter.index_types_[0] == pd.RangeIndex
        assert splitter.index_freqs_[0] == 1

    def test_init_single_long_format(self, df_datetime_long):
        """Test initialization with long format DataFrame with MultiIndex."""
        splitter = TimeSeriesSplitter(df_datetime_long)

        assert splitter.n_groups_ == 1
        assert splitter.n_timeseries == 2
        assert splitter.index_types_[0] == pd.DatetimeIndex

    def test_init_multiple_groups(self, multiple_dfs):
        """Test initialization with multiple series groups."""
        df1, df2, df3 = multiple_dfs
        splitter = TimeSeriesSplitter(df1, df2, df3)

        assert splitter.n_groups_ == 3
        assert splitter.n_timeseries == 3
        assert len(splitter.series_groups_) == 3
        assert len(splitter.index_types_) == 3
        assert all(idx_type == pd.DatetimeIndex for idx_type in splitter.index_types_)

    def test_init_empty_series_group_raises_error(self):
        """Test that empty series group raises ValueError."""
        # Create an empty DataFrame
        empty_df = pd.DataFrame()
        empty_dict = {}

        with pytest.raises(
            ValueError,
            match='If `series` is a dictionary, all series must have a Pandas RangeIndex or DatetimeIndex with the same step/frequency.',
        ):
            TimeSeriesSplitter(empty_df)

        with pytest.raises(
            ValueError,
            match='If `series` is a dictionary, all series must have a Pandas RangeIndex or DatetimeIndex with the same step/frequency.',
        ):
            TimeSeriesSplitter(empty_dict)

    def test_init_unsupported_index_type_raises_error(self):
        """Test that unsupported index type raises TypeError."""
        # Create DataFrame with unsupported index type (e.g., string index)
        df = pd.DataFrame(
            {'a': range(10)},
            index=pd.Index(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
        )

        with pytest.raises(
            TypeError,
            match='`series` has an unsupported index type. The index must be a pandas DatetimeIndex or a RangeIndex.',
        ):
            TimeSeriesSplitter(df)

    def test_init_datetime_index_frequency_stored(self, df_datetime_wide):
        """Test that DatetimeIndex frequency is correctly stored."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        assert splitter.index_freqs_[0] is not None
        assert hasattr(splitter.index_freqs_[0], 'freqstr')

    def test_init_min_max_indexes_datetime(self, df_datetime_wide):
        """Test min and max indexes are correctly computed for DatetimeIndex."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        assert splitter._min_indexes_[0] == pd.Timestamp('2023-01-01')
        assert splitter._max_indexes_[0] == pd.Timestamp('2023-04-10')

    def test_init_range_index_step_stored(self, df_range_wide):
        """Test that RangeIndex step is correctly stored."""
        splitter = TimeSeriesSplitter(df_range_wide)

        assert splitter.index_freqs_[0] == 1


class TestReprHtml:
    """Test the _repr_html_ method."""

    def test_repr_html_single_group_datetime(self, df_datetime_wide):
        """Test HTML representation for single group with DatetimeIndex."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        html = splitter._repr_html_()

        assert isinstance(html, str)
        assert '<style>' in html
        assert 'TimeSeriesSplitter' in html
        assert '<strong>Number of groups:</strong> 1' in html
        assert '<strong>Number of timeseries:</strong> 3' in html
        assert 'DatetimeIndex' in html
        assert 'Frequency:' in html

    def test_repr_html_single_group_range(self, df_range_wide):
        """Test HTML representation for single group with RangeIndex."""
        splitter = TimeSeriesSplitter(df_range_wide)
        html = splitter._repr_html_()

        assert 'RangeIndex' in html
        assert 'Step:' in html

    def test_repr_html_multiple_groups(self, multiple_dfs):
        """Test HTML representation for multiple groups."""
        df1, df2, df3 = multiple_dfs
        splitter = TimeSeriesSplitter(df1, df2, df3)
        html = splitter._repr_html_()

        assert '<strong>Number of groups:</strong> 3' in html
        assert 'Group 0:' in html
        assert 'Group 1:' in html
        assert 'Group 2:' in html

    def test_repr_html_contains_version_info(self, df_datetime_wide):
        """Test that HTML representation contains version information."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        html = splitter._repr_html_()

        assert 'Skforecast version:' in html
        assert 'Python version:' in html

    def test_repr_html_unique_id(self, df_datetime_wide):
        """Test that each HTML representation has a unique ID."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        html1 = splitter._repr_html_()
        html2 = splitter._repr_html_()

        # Extract unique IDs (they should be different)
        import re

        ids1 = re.findall(r'container-([a-f0-9]+)', html1)
        ids2 = re.findall(r'container-([a-f0-9]+)', html2)

        assert len(ids1) > 0
        assert len(ids2) > 0
        assert ids1[0] != ids2[0]


class TestConvertDateToPosition:
    """Test the _convert_date_to_position method."""

    def test_convert_date_to_position_string(self, df_datetime_wide):
        """Test converting string date to position."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        index = df_datetime_wide.index

        pos = splitter._convert_date_to_position('2023-01-01', index)
        assert pos == 0

        pos = splitter._convert_date_to_position('2023-01-15', index)
        assert pos == 14

    def test_convert_date_to_position_timestamp(self, df_datetime_wide):
        """Test converting Timestamp object to position."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        index = df_datetime_wide.index

        pos = splitter._convert_date_to_position(pd.Timestamp('2023-01-01'), index)
        assert pos == 0

    def test_convert_date_to_position_out_of_range_raises_error(self, df_datetime_wide):
        """Test that date outside range raises ValueError."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        index = df_datetime_wide.index

        with pytest.raises(ValueError, match='is not present in the series index'):
            splitter._convert_date_to_position('2022-12-31', index)

        with pytest.raises(ValueError, match='is not present in the series index'):
            splitter._convert_date_to_position('2023-05-01', index)

    def test_convert_date_to_position_custom_name(self, df_datetime_wide):
        """Test error message includes custom date name."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        index = df_datetime_wide.index

        with pytest.raises(ValueError, match='custom_date.*is not present'):
            splitter._convert_date_to_position(
                '2022-12-31', index, date_name='custom_date'
            )


class TestValidateDateSplitArgs:
    """Test the _validate_date_split_args method."""

    def test_validate_date_split_args_basic(self, df_datetime_wide):
        """Test basic date validation."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        start, end_train, end_val, end_test = splitter._validate_date_split_args(
            0, None, '2023-03-01', None, None
        )

        assert start == 0
        assert end_train == 59  # March 1st is 60th day from Jan 1st
        assert end_val == 59
        assert end_test == 99

    def test_validate_date_split_args_with_all_dates(self, df_datetime_wide):
        """Test validation with all dates specified."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        start, end_train, end_val, end_test = splitter._validate_date_split_args(
            0, '2023-01-10', '2023-02-01', '2023-03-01', '2023-03-20'
        )

        assert start == 9
        assert end_train == 31
        assert end_val == 59
        assert end_test == 78

    def test_validate_date_split_args_range_index_raises_error(self, df_range_wide):
        """Test that RangeIndex raises TypeError."""
        splitter = TimeSeriesSplitter(df_range_wide)

        with pytest.raises(TypeError, match='requires `DatetimeIndex`'):
            splitter._validate_date_split_args(0, None, '2023-01-01', None, None)

    def test_validate_date_split_args_start_after_end_raises_error(
        self, df_datetime_wide
    ):
        """Test that start_train >= end_train raises ValueError."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        with pytest.raises(
            ValueError, match='start_train must be earlier than end_train'
        ):
            splitter._validate_date_split_args(
                0, '2023-03-01', '2023-02-01', None, None
            )

    def test_validate_date_split_args_end_train_after_end_val_raises_error(
        self, df_datetime_wide
    ):
        """Test that end_train > end_validation raises ValueError."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        with pytest.raises(
            ValueError,
            match='end_train must be earlier than or equal to end_validation',
        ):
            splitter._validate_date_split_args(
                0, None, '2023-03-01', '2023-02-01', None
            )

    def test_validate_date_split_args_end_val_after_end_test_raises_error(
        self, df_datetime_wide
    ):
        """Test that end_validation > end_test raises ValueError."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        with pytest.raises(
            ValueError, match='end_validation must be earlier than or equal to end_test'
        ):
            splitter._validate_date_split_args(
                0, None, '2023-02-01', '2023-04-01', '2023-03-01'
            )

    def test_validate_date_split_args_negative_start_raises_error(
        self, df_datetime_wide
    ):
        """Test that start_train before series start raises ValueError."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        # This test is tricky because _convert_date_to_position will raise first
        # But if we somehow got a negative position, it should be caught
        with pytest.raises(ValueError, match='is not present in the series index'):
            splitter._validate_date_split_args(
                0, '2022-12-31', '2023-02-01', None, None
            )


class TestConvertSize:
    """Test the _convert_size method."""

    def test_convert_size_none_returns_none(self, df_datetime_wide):
        """Test that None size returns None."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        result = splitter._convert_size(None, 'test_size', 100)
        assert result is None

    def test_convert_size_integer_returns_integer(self, df_datetime_wide):
        """Test that integer size is returned as-is."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        result = splitter._convert_size(50, 'test_size', 100)
        assert result == 50

    def test_convert_size_float_converts_to_count(self, df_datetime_wide):
        """Test that float size is converted to count using ceiling."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        # 0.5 * 100 = 50
        result = splitter._convert_size(0.5, 'test_size', 100)
        assert result == 50

        # 0.33 * 100 = 33, ceil = 33
        result = splitter._convert_size(0.33, 'test_size', 100)
        assert result == 33

        # 0.01 * 100 = 1, ceil = 1
        result = splitter._convert_size(0.01, 'test_size', 100)
        assert result == 1

    def test_convert_size_float_out_of_range_raises_error(self, df_datetime_wide):
        """Test that float outside (0, 1) raises ValueError."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        with pytest.raises(ValueError, match='proportion must be between 0 and 1'):
            splitter._convert_size(0.0, 'test_size', 100)

        with pytest.raises(ValueError, match='proportion must be between 0 and 1'):
            splitter._convert_size(1.0, 'test_size', 100)

        with pytest.raises(ValueError, match='proportion must be between 0 and 1'):
            splitter._convert_size(1.5, 'test_size', 100)

        with pytest.raises(ValueError, match='proportion must be between 0 and 1'):
            splitter._convert_size(-0.5, 'test_size', 100)

    def test_convert_size_with_group_idx(self, df_datetime_wide):
        """Test that group_idx is included in error messages."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        with pytest.raises(ValueError, match='Group 5:'):
            splitter._convert_size(1.5, 'test_size', 100, group_idx=5)


class TestValidateSizeSplitArgs:
    """Test the _validate_size_split_args method."""

    def test_validate_size_split_args_basic(self, df_datetime_wide):
        """Test basic size validation."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, val, test = splitter._validate_size_split_args(0, 70, None, None)
        assert train == 70
        assert val is None
        assert test is None

    def test_validate_size_split_args_with_validation(self, df_datetime_wide):
        """Test size validation with validation set."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, val, test = splitter._validate_size_split_args(0, 60, 20, 20)
        assert train == 60
        assert val == 20
        assert test == 20

    def test_validate_size_split_args_proportions(self, df_datetime_wide):
        """Test size validation with proportions."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, val, test = splitter._validate_size_split_args(0, 0.6, 0.2, 0.2)
        assert train == 60
        assert val == 20
        assert test == 20

    def test_validate_size_split_args_exceeds_length_raises_error(
        self, df_datetime_wide
    ):
        """Test that total size exceeding length raises ValueError."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        with pytest.raises(ValueError, match='exceeds series length'):
            splitter._validate_size_split_args(0, 60, 30, 30)

    def test_validate_size_split_args_mixed_sizes(self, df_datetime_wide):
        """Test mixed integer and float sizes."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, val, test = splitter._validate_size_split_args(0, 50, 0.2, None)
        assert train == 50
        assert val == 20


class TestSplitSeriesDict:
    """Test the _split_series_dict method."""

    def test_split_series_dict_two_way(self, df_datetime_wide):
        """Test splitting into train and test."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        series_dict = splitter.series_groups_[0]

        positions = {'train': (0, 69), 'test': (70, 99)}

        splits = splitter._split_series_dict(series_dict, positions)

        assert len(splits) == 2
        assert 'series_a' in splits[0]
        assert len(splits[0]['series_a']) == 70
        assert len(splits[1]['series_a']) == 30

    def test_split_series_dict_three_way(self, df_datetime_wide):
        """Test splitting into train, validation, and test."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        series_dict = splitter.series_groups_[0]

        positions = {'train': (0, 59), 'validation': (60, 79), 'test': (80, 99)}

        splits = splitter._split_series_dict(series_dict, positions)

        assert len(splits) == 3
        assert len(splits[0]['series_a']) == 60
        assert len(splits[1]['series_a']) == 20
        assert len(splits[2]['series_a']) == 20

    def test_split_series_dict_preserves_data(self, df_datetime_wide):
        """Test that splitting preserves data correctly."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        series_dict = splitter.series_groups_[0]

        positions = {'train': (0, 49), 'test': (50, 99)}

        splits = splitter._split_series_dict(series_dict, positions)

        # Check that data is correctly preserved
        original_series_a = series_dict['series_a']
        assert splits[0]['series_a'].iloc[0] == original_series_a.iloc[0]
        assert splits[1]['series_a'].iloc[0] == original_series_a.iloc[50]


class TestConvertOutput:
    """Test the _convert_output method."""

    def test_convert_output_dict_format(self, df_datetime_wide):
        """Test converting to dict format."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        series_dict = splitter.series_groups_[0]

        positions = {'train': (0, 69), 'test': (70, 99)}
        splits = splitter._split_series_dict(series_dict, positions)

        result = splitter._convert_output(splits, 'dict')

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert 'series_a' in result[0]

    def test_convert_output_wide_format(self, df_datetime_wide):
        """Test converting to wide format."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        series_dict = splitter.series_groups_[0]

        positions = {'train': (0, 69), 'test': (70, 99)}
        splits = splitter._split_series_dict(series_dict, positions)

        result = splitter._convert_output(splits, 'wide')

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert 'series_a' in result[0].columns

    def test_convert_output_long_format(self, df_datetime_wide):
        """Test converting to long format."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        series_dict = splitter.series_groups_[0]

        positions = {'train': (0, 69), 'test': (70, 99)}
        splits = splitter._split_series_dict(series_dict, positions)

        result = splitter._convert_output(splits, 'long')

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)

    @pytest.mark.parametrize('freq', ['D', 'h'])
    def test_convert_output_long_format_series_with_different_spans(self, freq):
        """
        Test converting series with different spans to long format: the dates
        between their spans are added with NaN, in the frequency of the series.
        """
        series = {
            'series_a': pd.Series(
                range(10), index=pd.date_range('2023-01-01', periods=10, freq=freq),
                dtype=float
            ),
            'series_b': pd.Series(
                range(10), index=pd.date_range('2023-01-01', periods=20, freq=freq)[10:],
                dtype=float
            ),
        }
        splitter = TimeSeriesSplitter(series)
        positions = {'train': (0, 6), 'test': (7, 9)}
        splits = splitter._split_series_dict(splitter.series_groups_[0], positions)

        train, test = splitter._convert_output(splits, 'long_multi_index')

        expected_train_index = pd.date_range('2023-01-01', periods=17, freq=freq)
        for long in (train, test):
            for series_id in ('series_a', 'series_b'):
                assert long.loc[series_id].index.freq == pd.tseries.frequencies.to_offset(freq)
        pd.testing.assert_index_equal(
            train.loc['series_a'].index, expected_train_index, check_names=False
        )
        assert train['value'].notna().sum() == 14
        assert test['value'].notna().sum() == 6

    def test_convert_output_long_multi_index_format(self, df_datetime_wide):
        """Test converting to long multi-index format."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        series_dict = splitter.series_groups_[0]

        positions = {'train': (0, 69), 'test': (70, 99)}
        splits = splitter._split_series_dict(series_dict, positions)

        result = splitter._convert_output(splits, 'long_multi_index')

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)

    def test_unvalid_output_format(self, df_datetime_wide):
        """Test unsupported output format name."""
        splitter = TimeSeriesSplitter(df_datetime_wide)
        series_dict = splitter.series_groups_[0]

        positions = {'train': (0, 69), 'test': (70, 99)}
        splits = splitter._split_series_dict(series_dict, positions)

        with pytest.raises(
            ValueError, match='Output format `unvalid_format` is not supported.'
        ):
            splitter._convert_output(splits, 'unvalid_format')


class TestSplitByDate:
    """Test the split_by_date method."""

    def test_split_by_date_two_way_single_group(self, df_datetime_wide):
        """Test two-way split by date for single group."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, test = splitter.split_by_date(end_train='2023-03-11')

        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert len(train) == 70
        assert len(test) == 30

    def test_split_by_date_three_way_single_group(self, df_datetime_wide):
        """Test three-way split by date for single group."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, val, test = splitter.split_by_date(
            end_train='2023-02-19', end_validation='2023-03-11'
        )

        assert len(train) == 50
        assert len(val) == 20
        assert len(test) == 30

    def test_split_by_date_custom_start(self, df_datetime_wide):
        """Test split with custom start_train."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, test = splitter.split_by_date(
            start_train='2023-01-11', end_train='2023-03-11'
        )

        assert len(train) == 60
        assert train.index[0] == pd.Timestamp('2023-01-11')

    def test_split_by_date_custom_end_test(self, df_datetime_wide):
        """Test split with custom end_test."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, test = splitter.split_by_date(
            end_train='2023-02-19', end_test='2023-03-20'
        )

        assert len(test) == 29
        assert test.index[-1] == pd.Timestamp('2023-03-20')

    def test_split_by_date_dict_output(self, df_datetime_wide):
        """Test split with dict output format."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, test = splitter.split_by_date(
            end_train='2023-03-11', output_format='dict'
        )

        assert isinstance(train, dict)
        assert isinstance(test, dict)
        assert 'series_a' in train

    def test_split_by_date_long_output(self, df_datetime_wide):
        """Test split with long output format."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, test = splitter.split_by_date(
            end_train='2023-03-11', output_format='long'
        )

        assert isinstance(train, pd.DataFrame)
        assert 'datetime' in train.columns or isinstance(train.index, pd.MultiIndex)

    def test_split_by_date_long_multi_index_output(self, df_datetime_wide):
        """Test split with long_multi_index output format."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, test = splitter.split_by_date(
            end_train='2023-03-11', output_format='long_multi_index'
        )

        assert isinstance(train, pd.DataFrame)
        assert isinstance(train.index, pd.MultiIndex)

    def test_split_by_date_multiple_groups(self, multiple_dfs):
        """Test split by date for multiple groups."""
        df1, df2, df3 = multiple_dfs
        splitter = TimeSeriesSplitter(df1, df2, df3)

        results = splitter.split_by_date(end_train='2023-02-19')

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(len(r) == 2 for r in results)

    def test_split_by_date_verbose(self, df_datetime_wide, capsys):
        """Test split by date with verbose output."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        splitter.split_by_date(end_train='2023-03-11', verbose=True)

        captured = capsys.readouterr()
        assert 'Split Information' in captured.out
        assert 'Train' in captured.out
        assert 'Test' in captured.out

    def test_split_by_date_with_timestamp_objects(self, df_datetime_wide):
        """Test split using Timestamp objects instead of strings."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, test = splitter.split_by_date(end_train=pd.Timestamp('2023-03-11'))

        assert len(train) == 70
        assert len(test) == 30

    def test_split_by_date_removes_empty_series(self, df_datetime_wide_short):
        """Test that empty series are removed from splits."""
        splitter = TimeSeriesSplitter(df_datetime_wide_short)

        # Split so that test set is very small
        train, test = splitter.split_by_date(end_train='2023-01-09')

        # All series should still be present even if small
        assert 'series_a' in train.columns
        assert 'series_a' in test.columns


class TestSplitBySize:
    """Test the split_by_size method."""

    def test_split_by_size_two_way_absolute(self, df_datetime_wide):
        """Test two-way split by absolute size."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, test = splitter.split_by_size(train_size=70)

        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert len(train) == 70
        assert len(test) == 30

    def test_split_by_size_two_way_proportion(self, df_datetime_wide):
        """Test two-way split by proportion."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, test = splitter.split_by_size(train_size=0.7)

        assert len(train) == 70
        assert len(test) == 30

    def test_split_by_size_three_way_absolute(self, df_datetime_wide):
        """Test three-way split by absolute size."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, val, test = splitter.split_by_size(
            train_size=60, validation_size=20, test_size=20
        )

        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_split_by_size_three_way_proportion(self, df_datetime_wide):
        """Test three-way split by proportion."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, val, test = splitter.split_by_size(
            train_size=0.6, validation_size=0.2, test_size=0.2
        )

        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_split_by_size_mixed_sizes(self, df_datetime_wide):
        """Test split with mixed absolute and proportion sizes."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, val, test = splitter.split_by_size(
            train_size=60, validation_size=0.2, test_size=None
        )

        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_split_by_size_dict_output(self, df_datetime_wide):
        """Test split with dict output format."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, test = splitter.split_by_size(train_size=0.7, output_format='dict')

        assert isinstance(train, dict)
        assert isinstance(test, dict)

    def test_split_by_size_multiple_groups(self, multiple_dfs):
        """Test split by size for multiple groups."""
        df1, df2, df3 = multiple_dfs
        splitter = TimeSeriesSplitter(df1, df2, df3)

        results = splitter.split_by_size(train_size=0.7)

        assert isinstance(results, list)
        assert len(results) == 3

    def test_split_by_size_verbose(self, df_datetime_wide, capsys):
        """Test split by size with verbose output."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        splitter.split_by_size(train_size=0.7, verbose=True)

        captured = capsys.readouterr()
        assert 'Split Information' in captured.out

    def test_split_by_size_range_index(self, df_range_wide):
        """Test split by size works with RangeIndex."""
        splitter = TimeSeriesSplitter(df_range_wide)

        train, test = splitter.split_by_size(train_size=70)

        assert len(train) == 70
        assert len(test) == 30

    def test_split_by_size_no_validation(self, df_datetime_wide):
        """Test split without validation set."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        train, test = splitter.split_by_size(
            train_size=70, validation_size=None, test_size=30
        )

        assert len(train) == 70
        assert len(test) == 30

    @pytest.mark.parametrize('output_format', ['long', 'long_multi_index'])
    @pytest.mark.parametrize(
        'case', list(long_cases), ids=lambda case: f'case: {case}'
    )
    def test_split_by_size_long_output_series_with_shifted_dates(
        self, case, output_format
    ):
        """
        Test the long outputs keep all the values of series with the same
        frequency and shifted dates: each series has its own dates. Series
        whose dates are in the same grid are filled with NaN up to the span of
        the split, in the frequency of the series.
        """
        expected = long_cases[case]
        freq = expected['freq']
        periods = expected['periods']
        series = {
            'a': pd.Series(
                np.arange(periods, dtype=float),
                index=pd.date_range('2023-01-01', periods=periods, freq=freq),
            ),
            'b': pd.Series(
                np.arange(100, 100 + periods, dtype=float),
                index=pd.date_range(expected['start_b'], periods=periods, freq=freq),
            ),
        }
        splitter = TimeSeriesSplitter(series)
        return_multi_index = output_format == 'long_multi_index'

        train, test = splitter.split_by_size(
            train_size=expected['train_size'], output_format=output_format
        )

        pd.testing.assert_frame_equal(
            train, expected_long(expected['train'], return_multi_index)
        )
        pd.testing.assert_frame_equal(
            test, expected_long(expected['test'], return_multi_index)
        )

    @pytest.mark.parametrize('output_format', ['long', 'long_multi_index'])
    def test_split_by_size_long_output_hourly_series_with_timezone_and_gap(
        self, output_format
    ):
        """
        Test the long outputs keep all the values of hourly series with a
        timezone whose spans, of 9 hours, are separated by a gap: pandas gives
        the union of their dates a business hour frequency, and with it only
        the value at 09:00 was kept.
        """
        index = pd.date_range('2023-10-02', periods=35, freq='h', tz='Europe/Madrid')
        series = {
            'a': pd.Series(np.arange(10, dtype=float), index=index[:10]),
            'b': pd.Series(np.arange(100, 110, dtype=float), index=index[25:]),
        }
        splitter = TimeSeriesSplitter(series)

        train, _ = splitter.split_by_size(train_size=9, output_format=output_format)

        expected = pd.DataFrame(
            {
                'value': np.concatenate(
                    [
                        np.arange(9, dtype=float), np.full(25, np.nan),
                        np.full(25, np.nan), np.arange(100, 109, dtype=float),
                    ]
                )
            },
            index=pd.MultiIndex.from_product(
                [['a', 'b'], index[:34]], names=['series_id', 'datetime']
            ),
        )
        if output_format == 'long':
            expected = expected.reset_index()

        pd.testing.assert_frame_equal(train, expected)

    @pytest.mark.parametrize('output_format', ['long', 'long_multi_index'])
    def test_split_by_size_long_output_series_with_shifted_dates_and_timezone(
        self, output_format
    ):
        """
        Test the long outputs keep the dates and the values of daily series
        with a timezone and shifted dates when they contain a daylight saving
        change (2023-10-29 in Europe/Madrid).
        """
        series = {
            'a': pd.Series(
                np.arange(6, dtype=float),
                index=pd.date_range(
                    '2023-10-27', periods=6, freq='D', tz='Europe/Madrid'
                ),
            ),
            'b': pd.Series(
                np.arange(100, 106, dtype=float),
                index=pd.date_range(
                    '2023-10-27 06:00', periods=6, freq='D', tz='Europe/Madrid'
                ),
            ),
        }
        splitter = TimeSeriesSplitter(series)
        return_multi_index = output_format == 'long_multi_index'

        train, test = splitter.split_by_size(
            train_size=4, output_format=output_format
        )

        expected_train = expected_long(
            {
                'a': (
                    ['2023-10-27 00:00+02:00', '2023-10-28 00:00+02:00',
                     '2023-10-29 00:00+02:00', '2023-10-30 00:00+01:00'],
                    [0., 1., 2., 3.],
                ),
                'b': (
                    ['2023-10-27 06:00+02:00', '2023-10-28 06:00+02:00',
                     '2023-10-29 06:00+01:00', '2023-10-30 06:00+01:00'],
                    [100., 101., 102., 103.],
                ),
            },
            return_multi_index,
            tz='Europe/Madrid',
        )
        expected_test = expected_long(
            {
                'a': (['2023-10-31 00:00+01:00', '2023-11-01 00:00+01:00'], [4., 5.]),
                'b': (
                    ['2023-10-31 06:00+01:00', '2023-11-01 06:00+01:00'],
                    [104., 105.],
                ),
            },
            return_multi_index,
            tz='Europe/Madrid',
        )

        pd.testing.assert_frame_equal(train, expected_train)
        pd.testing.assert_frame_equal(test, expected_test)

    @pytest.mark.parametrize('output_format', ['long', 'long_multi_index'])
    def test_split_by_size_long_output_when_shared_grid_cannot_be_built(
        self, output_format
    ):
        """
        Test the long outputs have each series with its own dates when pandas
        cannot build the shared grid of the split: the daily grid that starts
        at 02:30 needs 2023-03-26 02:30, which does not exist in Europe/Madrid
        (daylight saving change).
        """
        series = {
            'b': pd.Series(
                np.arange(100, 105, dtype=float),
                index=pd.date_range(
                    '2023-03-20 02:30', periods=5, freq='D', tz='Europe/Madrid'
                ),
            ),
            'a': pd.Series(
                np.arange(5, dtype=float),
                index=pd.date_range(
                    '2023-03-24', periods=5, freq='D', tz='Europe/Madrid'
                ),
            ),
        }
        splitter = TimeSeriesSplitter(series)
        return_multi_index = output_format == 'long_multi_index'

        train, test = splitter.split_by_size(
            train_size=4, output_format=output_format
        )

        expected_train = expected_long(
            {
                'b': (
                    ['2023-03-20 02:30+01:00', '2023-03-21 02:30+01:00',
                     '2023-03-22 02:30+01:00', '2023-03-23 02:30+01:00'],
                    [100., 101., 102., 103.],
                ),
                'a': (
                    ['2023-03-24 00:00+01:00', '2023-03-25 00:00+01:00',
                     '2023-03-26 00:00+01:00', '2023-03-27 00:00+02:00'],
                    [0., 1., 2., 3.],
                ),
            },
            return_multi_index,
            tz='Europe/Madrid',
        )
        expected_test = expected_long(
            {
                'b': (['2023-03-24 02:30+01:00'], [104.]),
                'a': (['2023-03-28 00:00+02:00'], [4.]),
            },
            return_multi_index,
            tz='Europe/Madrid',
        )

        pd.testing.assert_frame_equal(train, expected_train)
        pd.testing.assert_frame_equal(test, expected_test)

    @pytest.mark.parametrize('output_format', ['long', 'long_multi_index'])
    def test_split_by_size_long_output_series_in_the_same_grid_and_shifted(
        self, output_format
    ):
        """
        Test the long outputs of a split with series in the same dates grid
        (`a` and `c`) and a series with shifted dates (`b`): each series has
        its own dates, in the order of the input and without NaN, the integer
        values are not converted to float and a series with no values in a
        split has no rows.
        """
        series = {
            'b': pd.Series(
                [2**53 + 1, 101, 102, 103, 104],
                index=pd.date_range('2023-01-04', periods=5, freq='7D'),
            ),
            'a': pd.Series(
                [0, 1, 2, 3], index=pd.date_range('2023-01-01', periods=4, freq='7D')
            ),
            'c': pd.Series(
                [200, 201, 202, 203, 204],
                index=pd.date_range('2023-01-15', periods=5, freq='7D'),
            ),
        }
        splitter = TimeSeriesSplitter(series)
        return_multi_index = output_format == 'long_multi_index'

        train, test = splitter.split_by_size(
            train_size=4, output_format=output_format
        )

        expected_train = expected_long(
            {
                'b': (
                    ['2023-01-04', '2023-01-11', '2023-01-18', '2023-01-25'],
                    [2**53 + 1, 101, 102, 103],
                ),
                'a': (
                    ['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22'],
                    [0, 1, 2, 3],
                ),
                'c': (
                    ['2023-01-15', '2023-01-22', '2023-01-29', '2023-02-05'],
                    [200, 201, 202, 203],
                ),
            },
            return_multi_index,
        )
        expected_test = expected_long(
            {'b': (['2023-02-01'], [104]), 'c': (['2023-02-12'], [204])},
            return_multi_index,
        )

        pd.testing.assert_frame_equal(train, expected_train)
        pd.testing.assert_frame_equal(test, expected_test)


class TestPrintSplitInfo:
    """Test the _print_split_info method."""

    def test_print_split_info_datetime_two_way(self, df_datetime_wide, capsys):
        """Test printing split info for DatetimeIndex with two-way split."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        positions = {'train': (0, 69), 'test': (70, 99)}

        splitter._print_split_info(0, positions, 'wide')

        captured = capsys.readouterr()
        assert 'Split Information' in captured.out
        assert 'Train' in captured.out
        assert 'Test' in captured.out
        assert '2023-01-01' in captured.out
        assert 'Output format: wide' in captured.out

    def test_print_split_info_datetime_three_way(self, df_datetime_wide, capsys):
        """Test printing split info for DatetimeIndex with three-way split."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        positions = {'train': (0, 59), 'validation': (60, 79), 'test': (80, 99)}

        splitter._print_split_info(0, positions, 'wide')

        captured = capsys.readouterr()
        assert 'Train' in captured.out
        assert 'Validation' in captured.out
        assert 'Test' in captured.out

    def test_print_split_info_range_index(self, df_range_wide, capsys):
        """Test printing split info for RangeIndex."""
        splitter = TimeSeriesSplitter(df_range_wide)

        positions = {'train': (0, 69), 'test': (70, 99)}

        splitter._print_split_info(0, positions, 'wide')

        captured = capsys.readouterr()
        assert 'Positions:' in captured.out
        assert '0 to 69' in captured.out

    def test_print_split_info_percentages(self, df_datetime_wide, capsys):
        """Test that split info includes correct percentages."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        positions = {'train': (0, 69), 'test': (70, 99)}

        splitter._print_split_info(0, positions, 'wide')

        captured = capsys.readouterr()
        assert '70.0%' in captured.out or '70%' in captured.out
        assert '30.0%' in captured.out or '30%' in captured.out

    def test_print_split_info_newline_for_multiple_groups(self, multiple_dfs, capsys):
        """Test that newline is printed between groups."""
        df1, df2, _ = multiple_dfs
        splitter = TimeSeriesSplitter(df1, df2)

        positions = {'train': (0, 49), 'test': (50, 99)}

        splitter._print_split_info(0, positions, 'wide')
        captured = capsys.readouterr()

        # Should have newline when group_idx < n_groups
        assert captured.out.endswith('\n\n')


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame(
            {'a': [1]}, index=pd.date_range('2023-01-01', periods=1, freq='D')
        )

        splitter = TimeSeriesSplitter(df)

        # Should not be able to split meaningfully
        with pytest.raises(ValueError):
            splitter.split_by_size(train_size=1, test_size=1)

    def test_very_small_dataframe(self):
        """Test handling of very small DataFrame (2 rows)."""
        df = pd.DataFrame(
            {'a': [1, 2]}, index=pd.date_range('2023-01-01', periods=2, freq='D')
        )

        splitter = TimeSeriesSplitter(df)
        train, test = splitter.split_by_size(train_size=1)

        assert len(train) == 1
        assert len(test) == 1

    def test_zero_length_split(self, df_datetime_wide):
        """Test handling when a split would have zero length."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        # All data to train, none to test (edge case)
        train, test = splitter.split_by_size(train_size=100, test_size=0)

        # Both should exist but test should be empty
        assert len(train) == 100

    def test_proportion_ceiling_behavior(self, df_datetime_wide):
        """Test that proportions use ceiling for size calculation."""
        splitter = TimeSeriesSplitter(df_datetime_wide)

        # 0.01 * 100 = 1, ceiling = 1
        result = splitter._convert_size(0.01, 'test', 100)
        assert result == 1

        # 0.999 * 100 = 99.9, ceiling = 100
        result = splitter._convert_size(0.999, 'test', 100)
        assert result == 100

    def test_different_series_lengths_in_dict(self):
        """Test that series with different lengths in dict are handled."""
        # This should work as each series has its own index
        series_dict = {
            'a': pd.Series(
                range(100), index=pd.date_range('2023-01-01', periods=100, freq='D')
            ),
            'b': pd.Series(
                range(100), index=pd.date_range('2023-01-01', periods=100, freq='D')
            ),
        }

        splitter = TimeSeriesSplitter(series_dict)
        assert splitter.n_timeseries == 2

    def test_multiindex_extraction(self, df_datetime_long):
        """Test proper extraction from MultiIndex DataFrame."""
        splitter = TimeSeriesSplitter(df_datetime_long)

        assert splitter.n_timeseries == 2
        assert len(splitter.series_groups_[0]) == 2
