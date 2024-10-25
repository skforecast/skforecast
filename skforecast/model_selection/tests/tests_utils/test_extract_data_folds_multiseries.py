# Unit test _extract_data_folds_multiseries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection._utils import _extract_data_folds_multiseries

# Fixtures
series = pd.DataFrame({
    'l1': np.arange(50, dtype=float),
    'l2': np.arange(50, 100, dtype=float),
    'l3': np.arange(100, 150, dtype=float)
})
exog = pd.DataFrame({
    'exog_1': np.arange(1000, 1050, dtype=float),
    'exog_2': np.arange(1050, 1100, dtype=float),
    'exog_3': np.arange(1100, 1150, dtype=float)
})


@pytest.mark.parametrize("dropna_last_window", 
                         [True, False], 
                         ids=lambda dropna: f'dropna_last_window: {dropna}')
def test_extract_data_folds_multiseries_series_DataFrame_exog_None_RangeIndex(dropna_last_window):
    """
    Test _extract_data_folds_multiseries with series DataFrame and exog None.
    """

    # Train, last_window, test_no_gap
    folds = [
        [[0, 30], [25, 30], [30, 37]], 
        [[0, 35], [30, 35], [35, 42]]
    ]
    span_index = pd.RangeIndex(start=0, stop=50, step=1)
    window_size = 5

    data_folds = list(
        _extract_data_folds_multiseries(
            series             = series, 
            folds              = folds, 
            span_index         = span_index, 
            window_size        = window_size, 
            exog               = None, 
            dropna_last_window = dropna_last_window, 
            externally_fitted  = False
        )
    )

    expected_data_folds = [
        (
            pd.DataFrame(
                data = {'l1': np.arange(0, 30, dtype=float),
                        'l2': np.arange(50, 80, dtype=float),
                        'l3': np.arange(100, 130, dtype=float)
                },
                index = pd.RangeIndex(start=0, stop=30, step=1)
            ), 
            pd.DataFrame(
                data = {'l1': np.arange(25, 30, dtype=float),
                        'l2': np.arange(75, 80, dtype=float),
                        'l3': np.arange(125, 130, dtype=float)
                },
                index = pd.RangeIndex(start=25, stop=30, step=1)
            ),
            ['l1', 'l2', 'l3'],
            None, 
            None, 
            folds[0]
        ),
        (
            pd.DataFrame(
                data = {'l1': np.arange(0, 35, dtype=float),
                        'l2': np.arange(50, 85, dtype=float),
                        'l3': np.arange(100, 135, dtype=float)
                },
                index = pd.RangeIndex(start=0, stop=35, step=1)
            ), 
            pd.DataFrame(
                data = {'l1': np.arange(30, 35, dtype=float),
                        'l2': np.arange(80, 85, dtype=float),
                        'l3': np.arange(130, 135, dtype=float)
                },
                index = pd.RangeIndex(start=30, stop=35, step=1)
            ),
            ['l1', 'l2', 'l3'],
            None,
            None,
            folds[1]
        )
    ]

    for i, data_fold in enumerate(data_folds):

        assert isinstance(data_fold, tuple)
        assert len(data_fold) == 6
        
        pd.testing.assert_frame_equal(data_fold[0], expected_data_folds[i][0])
        pd.testing.assert_frame_equal(data_fold[1], expected_data_folds[i][1])
        assert data_fold[2] == expected_data_folds[i][2]
        assert data_fold[3] is None
        assert data_fold[4] is None
        assert data_fold[5] == expected_data_folds[i][5]


@pytest.mark.parametrize("dropna_last_window", 
                         [True, False], 
                         ids=lambda dropna: f'dropna_last_window: {dropna}')
def test_extract_data_folds_multiseries_series_DataFrame_with_NaN_exog_RangeIndex(dropna_last_window):
    """
    Test _extract_data_folds_multiseries with series DataFrame and exog None with
    RangeIndex.
    """
    series_nan = series.copy()
    series_nan.loc[:28, 'l2'] = np.nan
    series_nan['l3'] = np.nan

    # Train, last_window, test_no_gap
    folds = [
        [[0, 30], [25, 30], [30, 37]], 
        [[0, 35], [30, 35], [35, 42]]
    ]
    span_index = pd.RangeIndex(start=0, stop=50, step=1)
    window_size = 5

    data_folds = list(
        _extract_data_folds_multiseries(
            series             = series_nan, 
            folds              = folds, 
            span_index         = span_index, 
            window_size        = window_size, 
            exog               = exog, 
            dropna_last_window = dropna_last_window, 
            externally_fitted  = False
        )
    )

    expected_data_folds = [
        (
            pd.DataFrame(
                data = {'l1': np.arange(0, 30, dtype=float)
                },
                index = pd.RangeIndex(start=0, stop=30, step=1)
            ),
            pd.DataFrame(
                data = {'l1': np.arange(25, 30, dtype=float)
                },
                index = pd.RangeIndex(start=25, stop=30, step=1)
            ),
            ['l1'],
            pd.DataFrame(
                data = {'exog_1': np.arange(1000, 1030, dtype=float),
                        'exog_2': np.arange(1050, 1080, dtype=float),
                        'exog_3': np.arange(1100, 1130, dtype=float)
                },
                index = pd.RangeIndex(start=0, stop=30, step=1)
            ),
            pd.DataFrame(
                data = {'exog_1': np.arange(1030, 1037, dtype=float),
                        'exog_2': np.arange(1080, 1087, dtype=float),
                        'exog_3': np.arange(1130, 1137, dtype=float)
                },
                index = pd.RangeIndex(start=30, stop=37, step=1)
            ),
            folds[0]
        ),
        (
            pd.DataFrame(
                data = {'l1': np.arange(0, 35, dtype=float),
                        'l2': [np.nan]*29 + list(range(79, 85))
                },
                index = pd.RangeIndex(start=0, stop=35, step=1)
            ).astype({'l2': float}), 
            pd.DataFrame(
                data = {'l1': np.arange(30, 35, dtype=float),
                        'l2': np.arange(80, 85, dtype=float)
                },
                index = pd.RangeIndex(start=30, stop=35, step=1)
            ),
            ['l1', 'l2'],
            pd.DataFrame(
                data = {'exog_1': np.arange(1000, 1035, dtype=float),
                        'exog_2': np.arange(1050, 1085, dtype=float),
                        'exog_3': np.arange(1100, 1135, dtype=float)
                },
                index = pd.RangeIndex(start=0, stop=35, step=1)
            ),
            pd.DataFrame(
                data = {'exog_1': np.arange(1035, 1042, dtype=float),
                        'exog_2': np.arange(1085, 1092, dtype=float),
                        'exog_3': np.arange(1135, 1142, dtype=float)
                },
                index = pd.RangeIndex(start=35, stop=42, step=1)
            ),
            folds[1]
        )
    ]

    for i, data_fold in enumerate(data_folds):

        assert isinstance(data_fold, tuple)
        assert len(data_fold) == 6
        
        pd.testing.assert_frame_equal(data_fold[0], expected_data_folds[i][0])
        pd.testing.assert_frame_equal(data_fold[1], expected_data_folds[i][1])
        assert data_fold[2] == expected_data_folds[i][2]
        pd.testing.assert_frame_equal(data_fold[3], expected_data_folds[i][3])
        pd.testing.assert_frame_equal(data_fold[4], expected_data_folds[i][4])
        assert data_fold[5] == expected_data_folds[i][5]


def test_extract_data_folds_multiseries_series_DataFrame_with_NaN_exog_DatetimeIndex():
    """
    Test _extract_data_folds_multiseries with series DataFrame with NaNs and exog 
    with DatetimeIndex.
    """
    series_nan = series.copy()
    series_nan.loc[28, 'l2'] = np.nan # 29th value of 'l2' is NaN (affected by window_size=5)
    series_nan.loc[:15, 'l3'] = np.nan # First 16 values of 'l3' are NaN

    series_nan.index = pd.date_range(start='2020-01-01', periods=50, freq='D')
    exog_datetime = exog.copy()
    exog_datetime.index = pd.date_range(start='2020-01-01', periods=50, freq='D')

    # Train, last_window, test_no_gap
    folds = [
        [[0, 30], [25, 30], [30, 37]], 
        [[0, 35], [30, 35], [35, 42]]
    ]
    span_index = pd.date_range(start='2020-01-01', periods=50, freq='D')
    window_size = 5

    data_folds = list(
        _extract_data_folds_multiseries(
            series             = series_nan, 
            folds              = folds, 
            span_index         = span_index, 
            window_size        = window_size, 
            exog               = exog_datetime, 
            dropna_last_window = True, 
            externally_fitted  = False
        )
    )

    expected_data_folds = [
        (
            pd.DataFrame(
                data = {'l1': np.arange(0, 30, dtype=float),
                        'l2': list(range(50, 78)) + [np.nan] + [79],
                        'l3': [np.nan]*16 + list(range(116, 130))
                },
                index = pd.date_range(start='2020-01-01', periods=30, freq='D')
            ).astype({'l2': float, 'l3': float}),
            pd.DataFrame(
                data = {'l1': np.arange(25, 30, dtype=float),
                        'l3': np.arange(125, 130, dtype=float)
                },
                index = pd.date_range(start='2020-01-26', periods=5, freq='D')
            ),
            ['l1', 'l3'],
            pd.DataFrame(
                data = {'exog_1': np.arange(1000, 1030, dtype=float),
                        'exog_2': np.arange(1050, 1080, dtype=float),
                        'exog_3': np.arange(1100, 1130, dtype=float)
                },
                index = pd.date_range(start='2020-01-01', periods=30, freq='D')
            ),
            pd.DataFrame(
                data = {'exog_1': np.arange(1030, 1037, dtype=float),
                        'exog_2': np.arange(1080, 1087, dtype=float),
                        'exog_3': np.arange(1130, 1137, dtype=float)
                },
                index = pd.date_range(start='2020-01-31', periods=7, freq='D')
            ),
            folds[0]
        ),
        (
            pd.DataFrame(
                data = {'l1': np.arange(0, 35, dtype=float),
                        'l2': list(range(50, 78)) + [np.nan] + list(range(79, 85)),
                        'l3': [np.nan]*16 + list(range(116, 135))
                },
                index = pd.date_range(start='2020-01-01', periods=35, freq='D')
            ).astype({'l2': float}), 
            pd.DataFrame(
                data = {'l1': np.arange(30, 35, dtype=float),
                        'l2': np.arange(80, 85, dtype=float),
                        'l3': np.arange(130, 135, dtype=float)
                },
                index = pd.date_range(start='2020-01-31', periods=5, freq='D')
            ),
            ['l1', 'l2', 'l3'],
            pd.DataFrame(
                data = {'exog_1': np.arange(1000, 1035, dtype=float),
                        'exog_2': np.arange(1050, 1085, dtype=float),
                        'exog_3': np.arange(1100, 1135, dtype=float)
                },
                index = pd.date_range(start='2020-01-01', periods=35, freq='D')
            ),
            pd.DataFrame(
                data = {'exog_1': np.arange(1035, 1042, dtype=float),
                        'exog_2': np.arange(1085, 1092, dtype=float),
                        'exog_3': np.arange(1135, 1142, dtype=float)
                },
                index = pd.date_range(start='2020-02-05', periods=7, freq='D')
            ),
            folds[1]
        )
    ]

    for i, data_fold in enumerate(data_folds):

        assert isinstance(data_fold, tuple)
        assert len(data_fold) == 6
        
        pd.testing.assert_frame_equal(data_fold[0], expected_data_folds[i][0])
        pd.testing.assert_frame_equal(data_fold[1], expected_data_folds[i][1])
        assert data_fold[2] == expected_data_folds[i][2]
        pd.testing.assert_frame_equal(data_fold[3], expected_data_folds[i][3])
        pd.testing.assert_frame_equal(data_fold[4], expected_data_folds[i][4])
        assert data_fold[5] == expected_data_folds[i][5]


@pytest.mark.parametrize("dropna_last_window", 
                         [True, False], 
                         ids=lambda dropna: f'dropna_last_window: {dropna}')
def test_extract_data_folds_multiseries_series_dict_exog_dict_DatetimeIndex(dropna_last_window):
    """
    Test _extract_data_folds_multiseries with series DataFrame and exog None with
    DatetimeIndex.
    """
    series_datetime = series.copy()
    series_datetime.index = pd.date_range(start='2020-01-01', periods=50, freq='D')

    series_dict = {
        'l1': series_datetime['l1'],
        'l2': series_datetime['l2'].iloc[10:],
        'l3': series_datetime['l3'].iloc[:20]
    }

    exog_datetime = exog.copy()
    exog_datetime.index = pd.date_range(start='2020-01-01', periods=50, freq='D')

    exog_dict = {
        'l1': exog_datetime,
        'l2': exog_datetime[['exog_1', 'exog_2']].iloc[15:],
        'l3': exog_datetime['exog_3'].iloc[30:]
    }

    # Train, last_window, test_no_gap
    folds = [
        [[0, 30], [25, 30], [30, 37]], 
        [[0, 35], [30, 35], [35, 42]]
    ]
    span_index = pd.date_range(start='2020-01-01', periods=50, freq='D')
    window_size = 5

    data_folds = list(
        _extract_data_folds_multiseries(
            series             = series_dict, 
            folds              = folds, 
            span_index         = span_index, 
            window_size        = window_size, 
            exog               = exog_dict, 
            dropna_last_window = dropna_last_window, 
            externally_fitted  = False
        )
    )

    expected_data_folds = [
        (
            {
                'l1': pd.Series(
                          data  = np.arange(0, 30, dtype=float), 
                          index = pd.date_range(start='2020-01-01', periods=30, freq='D'),
                          name  = 'l1'
                      ),
                'l2': pd.Series(
                          data  = np.arange(60, 80, dtype=float), 
                          index = pd.date_range(start='2020-01-11', periods=20, freq='D'),
                          name  = 'l2'
                      ),
                'l3': pd.Series(
                          data  = np.arange(100, 120, dtype=float), 
                          index = pd.date_range(start='2020-01-01', periods=20, freq='D'),
                          name  = 'l3'
                      )
            },
            pd.DataFrame(
                data = {'l1': np.arange(25, 30, dtype=float),
                        'l2': np.arange(75, 80, dtype=float)
                },
                index = pd.date_range(start='2020-01-26', periods=5, freq='D')
            ),
            ['l1', 'l2'],
            {
                'l1': pd.DataFrame(
                          data  = {'exog_1': np.arange(1000, 1030, dtype=float),
                                   'exog_2': np.arange(1050, 1080, dtype=float),
                                   'exog_3': np.arange(1100, 1130, dtype=float)}, 
                          index = pd.date_range(start='2020-01-01', periods=30, freq='D')
                      ),
                'l2': pd.DataFrame(
                          data  = {'exog_1': np.arange(1015, 1030, dtype=float),
                                   'exog_2': np.arange(1065, 1080, dtype=float)}, 
                          index = pd.date_range(start='2020-01-16', periods=15, freq='D')
                      )
            },
            {
                'l1': pd.DataFrame(
                          data  = {'exog_1': np.arange(1030, 1037, dtype=float),
                                   'exog_2': np.arange(1080, 1087, dtype=float),
                                   'exog_3': np.arange(1130, 1137, dtype=float)}, 
                          index = pd.date_range(start='2020-01-31', periods=7, freq='D')
                      ),
                'l2': pd.DataFrame(
                          data  = {'exog_1': np.arange(1030, 1037, dtype=float),
                                   'exog_2': np.arange(1080, 1087, dtype=float)}, 
                          index = pd.date_range(start='2020-01-31', periods=7, freq='D')
                      )
            },
            folds[0]
        ),
        (
            {
                'l1': pd.Series(
                          data  = np.arange(0, 35, dtype=float), 
                          index = pd.date_range(start='2020-01-01', periods=35, freq='D'),
                          name  = 'l1'
                      ),
                'l2': pd.Series(
                          data  = np.arange(60, 85, dtype=float), 
                          index = pd.date_range(start='2020-01-11', periods=25, freq='D'),
                          name  = 'l2'
                      ),
                'l3': pd.Series(
                          data  = np.arange(100, 120, dtype=float), 
                          index = pd.date_range(start='2020-01-01', periods=20, freq='D'),
                          name  = 'l3'
                      )
            }, 
            pd.DataFrame(
                data = {'l1': np.arange(30, 35, dtype=float),
                        'l2': np.arange(80, 85, dtype=float)
                },
                index = pd.date_range(start='2020-01-31', periods=5, freq='D')
            ),
            ['l1', 'l2'],
            {
                'l1': pd.DataFrame(
                          data  = {'exog_1': np.arange(1000, 1035, dtype=float),
                                   'exog_2': np.arange(1050, 1085, dtype=float),
                                   'exog_3': np.arange(1100, 1135, dtype=float)}, 
                          index = pd.date_range(start='2020-01-01', periods=35, freq='D')
                      ),
                'l2': pd.DataFrame(
                          data  = {'exog_1': np.arange(1015, 1035, dtype=float),
                                   'exog_2': np.arange(1065, 1085, dtype=float)}, 
                          index = pd.date_range(start='2020-01-16', periods=20, freq='D')
                      ),
                'l3': pd.Series(
                          data  = np.arange(1130, 1135, dtype=float), 
                          index = pd.date_range(start='2020-01-31', periods=5, freq='D'),
                          name  = 'exog_3'
                      )
            },
            {
                'l1': pd.DataFrame(
                          data  = {'exog_1': np.arange(1035, 1042, dtype=float),
                                   'exog_2': np.arange(1085, 1092, dtype=float),
                                   'exog_3': np.arange(1135, 1142, dtype=float)}, 
                          index = pd.date_range(start='2020-02-05', periods=7, freq='D')
                      ),
                'l2': pd.DataFrame(
                          data  = {'exog_1': np.arange(1035, 1042, dtype=float),
                                   'exog_2': np.arange(1085, 1092, dtype=float)}, 
                          index = pd.date_range(start='2020-02-05', periods=7, freq='D')
                      ),
                'l3': pd.Series(
                          data  = np.arange(1135, 1142, dtype=float), 
                          index = pd.date_range(start='2020-02-05', periods=7, freq='D'),
                          name  = 'exog_3'
                      )
            },
            folds[1]
        )
    ]

    for i, data_fold in enumerate(data_folds):

        assert isinstance(data_fold, tuple)
        assert len(data_fold) == 6
        
        for key in data_fold[0].keys():
            pd.testing.assert_series_equal(data_fold[0][key], expected_data_folds[i][0][key])
        pd.testing.assert_frame_equal(data_fold[1], expected_data_folds[i][1])

        assert data_fold[2] == expected_data_folds[i][2]

        for key in data_fold[3].keys():
            if isinstance(expected_data_folds[i][3][key], pd.DataFrame):
                pd.testing.assert_frame_equal(data_fold[3][key], expected_data_folds[i][3][key])
            else:
                pd.testing.assert_series_equal(data_fold[3][key], expected_data_folds[i][3][key])
        for key in data_fold[4].keys():
            if isinstance(expected_data_folds[i][4][key], pd.DataFrame):
                pd.testing.assert_frame_equal(data_fold[4][key], expected_data_folds[i][4][key])
            else:
                pd.testing.assert_series_equal(data_fold[4][key], expected_data_folds[i][4][key])

        assert data_fold[5] == expected_data_folds[i][5]