# Unit test TimeSeriesFold
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection._split import TimeSeriesFold


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(67, 70), range(70, 80), range(70, 80), True],
                                  [range(0, 70), range(77, 80), range(80, 90), range(80, 90), False],
                                  [range(0, 70), range(87, 90), range(90, 100), range(90, 100), False]]),
                          (False, [[[0, 70], [67, 70], [70, 80], [70, 80], True],
                                   [[0, 70], [77, 80], [80, 90], [80, 90], False],
                                   [[0, 70], [87, 90], [90, 100], [90, 100], False]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_no_refit_no_gap_no_remainder(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is False, gap=0 and not 
    remainder.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = TimeSeriesFold(
            steps                 = 10,
            initial_train_size    = 70,
            window_size           = 3,
            differentiation       = None,
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 0\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-12 00:00:00 -- 2022-03-21 00:00:00  (n=10)\n"
        "Fold: 1\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-03-22 00:00:00 -- 2022-03-31 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-04-01 00:00:00 -- 2022-04-10 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 65), range(61, 65), range(65, 75), range(65, 75), True],
                                  [range(0, 65), range(71, 75), range(75, 85), range(75, 85), False],
                                  [range(0, 65), range(81, 85), range(85, 95), range(85, 95), False]]),
                          (False, [[[0, 65], [61, 65], [65, 75], [65, 75], True],
                                   [[0, 65], [71, 75], [75, 85], [75, 85], False],
                                   [[0, 65], [81, 85], [85, 95], [85, 95], False]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_no_refit_no_gap_allow_incomplete_fold_False(capfd, return_all_indexes, expected):
    """s
    Test TimeSeriesFold split method output when refit is 0 (False), gap=0, 
    remainder and allow_incomplete_fold=False.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = TimeSeriesFold(
            steps                 = 10,
            initial_train_size    = 65,
            window_size           = 4,
            differentiation       = None,
            refit                 = 0,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = False,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 65\n"
        "Number of observations used for backtesting: 35\n"
        "    Number of folds: 3\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 0\n"
        "    Last fold has been excluded because it was incomplete.\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-06 00:00:00  (n=65)\n"
        "    Validation: 2022-03-07 00:00:00 -- 2022-03-16 00:00:00  (n=10)\n"
        "Fold: 1\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-03-17 00:00:00 -- 2022-03-26 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-03-27 00:00:00 -- 2022-04-05 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(65, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 70), range(72, 77), range(77, 89), range(82, 89), False],
                                  [range(0, 70), range(79, 84), range(84, 96), range(89, 96), False],
                                  [range(0, 70), range(86, 91), range(91, 100), range(96, 100), False]]),
                          (False, [[[0, 70], [65, 70], [70, 82], [75, 82], True],
                                   [[0, 70], [72, 77], [77, 89], [82, 89], False],
                                   [[0, 70], [79, 84], [84, 96], [89, 96], False],
                                   [[0, 70], [86, 91], [91, 100], [96, 100], False]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_no_refit_gap_allow_incomplete_fold_True(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is False, gap=5, 
    remainder, allow_incomplete_fold=True.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = TimeSeriesFold(
            steps                 = 7,
            initial_train_size    = 70,
            window_size           = 5,
            differentiation       = None,
            refit                 = False,
            fixed_train_size      = False,
            gap                   = 5,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 4\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold only includes 4 observations.\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-17 00:00:00 -- 2022-03-23 00:00:00  (n=7)\n"
        "Fold: 1\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-03-24 00:00:00 -- 2022-03-30 00:00:00  (n=7)\n"
        "Fold: 2\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-03-31 00:00:00 -- 2022-04-06 00:00:00  (n=7)\n"
        "Fold: 3\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-04-07 00:00:00 -- 2022-04-10 00:00:00  (n=4)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 15), range(0, 15), range(15, 30), range(20, 30), False],
                                  [range(0, 15), range(10, 25), range(25, 40), range(30, 40), False],
                                  [range(0, 15), range(20, 35), range(35, 50), range(40, 50), False],
                                  [range(0, 15), range(30, 45), range(45, 60), range(50, 60), False],
                                  [range(0, 15), range(40, 55), range(55, 70), range(60, 70), False],
                                  [range(0, 15), range(50, 65), range(65, 80), range(70, 80), False],
                                  [range(0, 15), range(60, 75), range(75, 90), range(80, 90), False],
                                  [range(0, 15), range(70, 85), range(85, 100), range(90, 100), False]]),
                          (False, [[[0, 15], [0, 15], [15, 30], [20, 30], False],
                                   [[0, 15], [10, 25], [25, 40], [30, 40], False],
                                   [[0, 15], [20, 35], [35, 50], [40, 50], False],
                                   [[0, 15], [30, 45], [45, 60], [50, 60], False],
                                   [[0, 15], [40, 55], [55, 70], [60, 70], False],
                                   [[0, 15], [50, 65], [65, 80], [70, 80], False],
                                   [[0, 15], [60, 75], [75, 90], [80, 90], False],
                                   [[0, 15], [70, 85], [85, 100], [90, 100], False]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_no_refit_initial_train_size_None_gap(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is False, gap=5, 
    initial_train_size=None (externally fitted), no remainder, allow_incomplete_fold=True.
    """
    y = pd.Series(np.arange(100))
    cv = TimeSeriesFold(
             steps                 = 10,
             initial_train_size    = None,
             window_size           = 15,
             differentiation       = None,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 5,
             skip_folds            = None,
             allow_incomplete_fold = True,
             return_all_indexes    = return_all_indexes,
         )
    folds = cv.split(X=y)            
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "An already trained forecaster is to be used. Window size: 15\n"
        "Number of observations used for backtesting: 85\n"
        "    Number of folds: 8\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n\n"
        "Fold: 0\n"
        "    Training:   No training in this fold\n"
        "    Validation: 20 -- 29  (n=10)\n"
        "Fold: 1\n"
        "    Training:   No training in this fold\n"
        "    Validation: 30 -- 39  (n=10)\n"
        "Fold: 2\n"
        "    Training:   No training in this fold\n"
        "    Validation: 40 -- 49  (n=10)\n"
        "Fold: 3\n"
        "    Training:   No training in this fold\n"
        "    Validation: 50 -- 59  (n=10)\n"
        "Fold: 4\n"
        "    Training:   No training in this fold\n"
        "    Validation: 60 -- 69  (n=10)\n"
        "Fold: 5\n"
        "    Training:   No training in this fold\n"
        "    Validation: 70 -- 79  (n=10)\n"
        "Fold: 6\n"
        "    Training:   No training in this fold\n"
        "    Validation: 80 -- 89  (n=10)\n"
        "Fold: 7\n"
        "    Training:   No training in this fold\n"
        "    Validation: 90 -- 99  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(68, 70), range(70, 80), range(70, 80), True],
                                  [range(0, 80), range(78, 80), range(80, 90), range(80, 90), True],
                                  [range(0, 90), range(88, 90), range(90, 100), range(90, 100), True]]),
                          (False, [[[0, 70], [68, 70], [70, 80], [70, 80], True],
                                   [[0, 80], [78, 80], [80, 90], [80, 90], True],
                                   [[0, 90], [88, 90], [90, 100], [90, 100], True]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_no_fixed_no_gap_no_remainder(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is True, fixed_train_size is 
    False, gap=0 and not remainder.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = TimeSeriesFold(
            steps                 = 10,
            initial_train_size    = 70,
            window_size           = 2,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = False,
            return_all_indexes    = return_all_indexes,
        )

    folds = cv.split(X=y)
    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 0\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-12 00:00:00 -- 2022-03-21 00:00:00  (n=10)\n"
        "Fold: 1\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-21 00:00:00  (n=80)\n"
        "    Validation: 2022-03-22 00:00:00 -- 2022-03-31 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-31 00:00:00  (n=90)\n"
        "    Validation: 2022-04-01 00:00:00 -- 2022-04-10 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(64, 70), range(70, 80), range(70, 80), True],
                                  [range(10, 80), range(74, 80), range(80, 90), range(80, 90), True],
                                  [range(20, 90), range(84, 90), range(90, 100), range(90, 100), True]]),
                          (False, [[[0, 70], [64, 70], [70, 80], [70, 80], True],
                                   [[10, 80], [74, 80], [80, 90], [80, 90], True],
                                   [[20, 90], [84, 90], [90, 100], [90, 100], True]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_fixed_train_size_no_gap_no_remainder(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is 1 (True), fixed_train_size 
    is True, gap=0 and not remainder.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = TimeSeriesFold(
            steps                 = 10,
            initial_train_size    = 70,
            window_size           = 6,
            differentiation       = None,
            refit                 = 1,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 0\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-12 00:00:00 -- 2022-03-21 00:00:00  (n=10)\n"
        "Fold: 1\n"
        "    Training:   2022-01-11 00:00:00 -- 2022-03-21 00:00:00  (n=70)\n"
        "    Validation: 2022-03-22 00:00:00 -- 2022-03-31 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   2022-01-21 00:00:00 -- 2022-03-31 00:00:00  (n=70)\n"
        "    Validation: 2022-04-01 00:00:00 -- 2022-04-10 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(67, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 77), range(74, 77), range(77, 89), range(82, 89), True],
                                  [range(0, 84), range(81, 84), range(84, 96), range(89, 96), True],
                                  [range(0, 91), range(88, 91), range(91, 100), range(96, 100), True]]),
                          (False, [[[0, 70], [67, 70], [70, 82], [75, 82], True],
                                   [[0, 77], [74, 77], [77, 89], [82, 89], True],
                                   [[0, 84], [81, 84], [84, 96], [89, 96], True],
                                   [[0, 91], [88, 91], [91, 100], [96, 100], True]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_no_fixed_gap_allow_incomplete_fold_True(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is True, fixed_train_size is 
    False, gap=5, remainder, allow_incomplete_fold=True.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = TimeSeriesFold(
            steps                 = 7,
            initial_train_size    = 70,
            window_size           = 3,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 5,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 4\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold only includes 4 observations.\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-17 00:00:00 -- 2022-03-23 00:00:00  (n=7)\n"
        "Fold: 1\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-18 00:00:00  (n=77)\n"
        "    Validation: 2022-03-24 00:00:00 -- 2022-03-30 00:00:00  (n=7)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-25 00:00:00  (n=84)\n"
        "    Validation: 2022-03-31 00:00:00 -- 2022-04-06 00:00:00  (n=7)\n"
        "Fold: 3\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-04-01 00:00:00  (n=91)\n"
        "    Validation: 2022-04-07 00:00:00 -- 2022-04-10 00:00:00  (n=4)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(65, 70), range(70, 82), range(75, 82), True],
                                  [range(7, 77), range(72, 77), range(77, 89), range(82, 89), True],
                                  [range(14, 84), range(79, 84), range(84, 96), range(89, 96), True],
                                  [range(21, 91), range(86, 91), range(91, 100), range(96, 100), True]]),
                          (False, [[[0, 70], [65, 70], [70, 82], [75, 82], True],
                                   [[7, 77], [72, 77], [77, 89], [82, 89], True],
                                   [[14, 84], [79, 84], [84, 96], [89, 96], True],
                                   [[21, 91], [86, 91], [91, 100], [96, 100], True]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_fixed_train_size_gap_allow_incomplete_fold_True(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is True, fixed_train_size is 
    True, gap=5, remainder, allow_incomplete_fold=True.
    """
    y = pd.Series(np.arange(100))
    cv = TimeSeriesFold(
            steps                 = 7,
            initial_train_size    = 70,
            window_size           = 5,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 5,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 4\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold only includes 4 observations.\n\n"
        "Fold: 0\n"
        "    Training:   0 -- 69  (n=70)\n"
        "    Validation: 75 -- 81  (n=7)\n"
        "Fold: 1\n"
        "    Training:   7 -- 76  (n=70)\n"
        "    Validation: 82 -- 88  (n=7)\n"
        "Fold: 2\n"
        "    Training:   14 -- 83  (n=70)\n"
        "    Validation: 89 -- 95  (n=7)\n"
        "Fold: 3\n"
        "    Training:   21 -- 90  (n=70)\n"
        "    Validation: 96 -- 99  (n=4)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(67, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 77), range(74, 77), range(77, 89), range(82, 89), True],
                                  [range(0, 84), range(81, 84), range(84, 96), range(89, 96), True]]),
                          (False, [[[0, 70], [67, 70], [70, 82], [75, 82], True],
                                   [[0, 77], [74, 77], [77, 89], [82, 89], True],
                                   [[0, 84], [81, 84], [84, 96], [89, 96], True]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_no_fixed_gap_allow_incomplete_fold_False(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is True, fixed_train_size is 
    False, gap=5, remainder, allow_incomplete_fold=False.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = TimeSeriesFold(
            steps                 = 7,
            initial_train_size    = 70,
            window_size           = 3,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 5,
            skip_folds            = None,
            allow_incomplete_fold = False,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold has been excluded because it was incomplete.\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-17 00:00:00 -- 2022-03-23 00:00:00  (n=7)\n"
        "Fold: 1\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-18 00:00:00  (n=77)\n"
        "    Validation: 2022-03-24 00:00:00 -- 2022-03-30 00:00:00  (n=7)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-25 00:00:00  (n=84)\n"
        "    Validation: 2022-03-31 00:00:00 -- 2022-04-06 00:00:00  (n=7)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(66, 70), range(70, 82), range(75, 82), True],
                                  [range(7, 77), range(73, 77), range(77, 89), range(82, 89), True],
                                  [range(14, 84), range(80, 84), range(84, 96), range(89, 96), True]]),
                          (False, [[[0, 70], [66, 70], [70, 82], [75, 82], True],
                                   [[7, 77], [73, 77], [77, 89], [82, 89], True],
                                   [[14, 84], [80, 84], [84, 96], [89, 96], True]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_fixed_train_size_gap_allow_incomplete_fold_False(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is True, fixed_train_size is 
    True, gap=5, remainder, allow_incomplete_fold=False.
    """
    y = pd.Series(np.arange(100))
    cv = TimeSeriesFold(
            steps                 = 7,
            initial_train_size    = 70,
            window_size           = 4,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 5,
            skip_folds            = None,
            allow_incomplete_fold = False,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold has been excluded because it was incomplete.\n\n"
        "Fold: 0\n"
        "    Training:   0 -- 69  (n=70)\n"
        "    Validation: 75 -- 81  (n=7)\n"
        "Fold: 1\n"
        "    Training:   7 -- 76  (n=70)\n"
        "    Validation: 82 -- 88  (n=7)\n"
        "Fold: 2\n"
        "    Training:   14 -- 83  (n=70)\n"
        "    Validation: 89 -- 95  (n=7)\n\n"
    )

    assert out == expected_out
    assert folds == expected


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 60), range(56, 60), range(60, 70), range(60, 70), True],
                                  [range(0, 60), range(66, 70), range(70, 80), range(70, 80), False],
                                  [range(0, 80), range(76, 80), range(80, 90), range(80, 90), True],
                                  [range(0, 80), range(86, 90), range(90, 100), range(90, 100), False]]),
                          (False, [[[0, 60], [56, 60], [60, 70], [60, 70], True],
                                   [[0, 60], [66, 70], [70, 80], [70, 80], False],
                                   [[0, 80], [76, 80], [80, 90], [80, 90], True],
                                   [[0, 80], [86, 90], [90, 100], [90, 100], False]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_int_no_fixed_no_gap_no_remainder(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is 2, fixed_train_size is 
    False, gap=0 and not remainder.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = TimeSeriesFold(
            steps                 = 10,
            initial_train_size    = 60,
            window_size           = 4,
            differentiation       = None,
            refit                 = 2,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = False,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 60\n"
        "Number of observations used for backtesting: 40\n"
        "    Number of folds: 4\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 0\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-01 00:00:00  (n=60)\n"
        "    Validation: 2022-03-02 00:00:00 -- 2022-03-11 00:00:00  (n=10)\n"
        "Fold: 1\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-03-12 00:00:00 -- 2022-03-21 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-21 00:00:00  (n=80)\n"
        "    Validation: 2022-03-22 00:00:00 -- 2022-03-31 00:00:00  (n=10)\n"
        "Fold: 3\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-04-01 00:00:00 -- 2022-04-10 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 60), range(58, 60), range(60, 70), range(60, 70), True],
                                  [range(0, 60), range(68, 70), range(70, 80), range(70, 80), False],
                                  [range(0, 60), range(78, 80), range(80, 90), range(80, 90), False],
                                  [range(30, 90), range(88, 90), range(90, 100), range(90, 100), True]]),
                          (False, [[[0, 60], [58, 60], [60, 70], [60, 70], True],
                                   [[0, 60], [68, 70], [70, 80], [70, 80], False],
                                   [[0, 60], [78, 80], [80, 90], [80, 90], False],
                                   [[30, 90], [88, 90], [90, 100], [90, 100], True]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_int_fixed_train_size_no_gap_no_remainder(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is 3, fixed_train_size is 
    True, gap=0 and not remainder.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = TimeSeriesFold(
            steps                 = 10,
            initial_train_size    = 60,
            window_size           = 2,
            differentiation       = None,
            refit                 = 3,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = False,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 60\n"
        "Number of observations used for backtesting: 40\n"
        "    Number of folds: 4\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 0\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-01 00:00:00  (n=60)\n"
        "    Validation: 2022-03-02 00:00:00 -- 2022-03-11 00:00:00  (n=10)\n"
        "Fold: 1\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-03-12 00:00:00 -- 2022-03-21 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-03-22 00:00:00 -- 2022-03-31 00:00:00  (n=10)\n"
        "Fold: 3\n"
        "    Training:   2022-01-31 00:00:00 -- 2022-03-31 00:00:00  (n=60)\n"
        "    Validation: 2022-04-01 00:00:00 -- 2022-04-10 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(60, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 70), range(67, 77), range(77, 89), range(82, 89), False],
                                  [range(0, 84), range(74, 84), range(84, 96), range(89, 96), True],
                                  [range(0, 84), range(81, 91), range(91, 100), range(96, 100), False]]),
                          (False, [[[0, 70], [60, 70], [70, 82], [75, 82], True],
                                   [[0, 70], [67, 77], [77, 89], [82, 89], False],
                                   [[0, 84], [74, 84], [84, 96], [89, 96], True],
                                   [[0, 84], [81, 91], [91, 100], [96, 100], False]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_int_no_fixed_gap_allow_incomplete_fold_True(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is 2, fixed_train_size is 
    False, gap=5, remainder, allow_incomplete_fold=True.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = TimeSeriesFold(
            steps                 = 7,
            initial_train_size    = 70,
            window_size           = 10,
            differentiation       = None,
            refit                 = 2,
            fixed_train_size      = False,
            gap                   = 5,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 4\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold only includes 4 observations.\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-17 00:00:00 -- 2022-03-23 00:00:00  (n=7)\n"
        "Fold: 1\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-03-24 00:00:00 -- 2022-03-30 00:00:00  (n=7)\n"
        "Fold: 2\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-25 00:00:00  (n=84)\n"
        "    Validation: 2022-03-31 00:00:00 -- 2022-04-06 00:00:00  (n=7)\n"
        "Fold: 3\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-04-07 00:00:00 -- 2022-04-10 00:00:00  (n=4)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(55, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 70), range(62, 77), range(77, 89), range(82, 89), False],
                                  [range(0, 70), range(69, 84), range(84, 96), range(89, 96), False],
                                  [range(21, 91), range(76, 91), range(91, 100), range(96, 100), True]]),
                          (False, [[[0, 70], [55, 70], [70, 82], [75, 82], True],
                                   [[0, 70], [62, 77], [77, 89], [82, 89], False],
                                   [[0, 70], [69, 84], [84, 96], [89, 96], False],
                                   [[21, 91], [76, 91], [91, 100], [96, 100], True]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_int_fixed_train_size_gap_allow_incomplete_fold_True(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is 3, fixed_train_size is 
    True, gap=5, remainder, allow_incomplete_fold=True.
    """
    y = pd.Series(np.arange(100))
    cv = TimeSeriesFold(
            steps                 = 7,
            initial_train_size    = 70,
            window_size           = 15,
            differentiation       = None,
            refit                 = 3,
            fixed_train_size      = True,
            gap                   = 5,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 4\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold only includes 4 observations.\n\n"
        "Fold: 0\n"
        "    Training:   0 -- 69  (n=70)\n"
        "    Validation: 75 -- 81  (n=7)\n"
        "Fold: 1\n"
        "    Training:   No training in this fold\n"
        "    Validation: 82 -- 88  (n=7)\n"
        "Fold: 2\n"
        "    Training:   No training in this fold\n"
        "    Validation: 89 -- 95  (n=7)\n"
        "Fold: 3\n"
        "    Training:   21 -- 90  (n=70)\n"
        "    Validation: 96 -- 99  (n=4)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    

@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(50, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 70), range(57, 77), range(77, 89), range(82, 89), False],
                                  [range(0, 70), range(64, 84), range(84, 96), range(89, 96), False]]),
                          (False, [[[0, 70], [50, 70], [70, 82], [75, 82], True],
                                   [[0, 70], [57, 77], [77, 89], [82, 89], False],
                                   [[0, 70], [64, 84], [84, 96], [89, 96], False]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_int_no_fixed_gap_allow_incomplete_fold_False(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is 3, fixed_train_size is 
    False, gap=5, remainder, allow_incomplete_fold=False.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = TimeSeriesFold(
            steps                 = 7,
            initial_train_size    = 70,
            window_size           = 20,
            differentiation       = None,
            refit                 = 3,
            fixed_train_size      = False,
            gap                   = 5,
            skip_folds            = None,
            allow_incomplete_fold = False,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold has been excluded because it was incomplete.\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-17 00:00:00 -- 2022-03-23 00:00:00  (n=7)\n"
        "Fold: 1\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-03-24 00:00:00 -- 2022-03-30 00:00:00  (n=7)\n"
        "Fold: 2\n"
        "    Training:   No training in this fold\n"
        "    Validation: 2022-03-31 00:00:00 -- 2022-04-06 00:00:00  (n=7)\n\n"
    )

    assert out == expected_out
    assert folds == expected


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(60, 70), range(70, 82), range(75, 82), True],
                                  [range(0, 70), range(67, 77), range(77, 89), range(82, 89), False],
                                  [range(14, 84), range(74, 84), range(84, 96), range(89, 96), True]]),
                          (False, [[[0, 70], [60, 70], [70, 82], [75, 82], True],
                                   [[0, 70], [67, 77], [77, 89], [82, 89], False],
                                   [[14, 84], [74, 84], [84, 96], [89, 96], True]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_int_fixed_train_size_gap_allow_incomplete_fold_False(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is True, fixed_train_size is 
    True, gap=5, remainder, allow_incomplete_fold=False.
    """
    y = pd.Series(np.arange(100))
    cv = TimeSeriesFold(
            steps                 = 7,
            initial_train_size    = 70,
            window_size           = 10,
            differentiation       = None,
            refit                 = 2,
            fixed_train_size      = True,
            gap                   = 5,
            skip_folds            = None,
            allow_incomplete_fold = False,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
                    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 7\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 5\n"
        "    Last fold has been excluded because it was incomplete.\n\n"
        "Fold: 0\n"
        "    Training:   0 -- 69  (n=70)\n"
        "    Validation: 75 -- 81  (n=7)\n"
        "Fold: 1\n"
        "    Training:   No training in this fold\n"
        "    Validation: 82 -- 88  (n=7)\n"
        "Fold: 2\n"
        "    Training:   14 -- 83  (n=70)\n"
        "    Validation: 89 -- 95  (n=7)\n\n"
    )

    assert out == expected_out
    assert folds == expected


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(67, 70), range(70, 80), range(70, 80), True],
                                  [range(0, 80), range(77, 80), range(80, 90), range(80, 90), True],
                                  [range(0, 90), range(87, 90), range(90, 100), range(90, 100), True]]),
                          (False, [[[0, 70], [67, 70], [70, 80], [70, 80], True],
                                   [[0, 80], [77, 80], [80, 90], [80, 90], True],
                                   [[0, 90], [87, 90], [90, 100], [90, 100], True]])], 
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_no_fixed_no_gap_no_remainder_differentiation(capfd, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is True, fixed_train_size is 
    False, gap=0, not remainder and differentiation=1.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    differentiation = 1
    cv = TimeSeriesFold(
            steps                 = 10,
            initial_train_size    = 70,
            window_size           = 2 + differentiation,
            differentiation       = differentiation,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)
    
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 69\n"
        "    First 1 observation/s in training sets are used for differentiation\n"
        "Number of observations used for backtesting: 30\n"
        "    Number of folds: 3\n"
        "    Number skipped folds: 0 \n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 0\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-02 00:00:00 -- 2022-03-11 00:00:00  (n=69)\n"
        "    Validation: 2022-03-12 00:00:00 -- 2022-03-21 00:00:00  (n=10)\n"
        "Fold: 1\n"
        "    Training:   2022-01-02 00:00:00 -- 2022-03-21 00:00:00  (n=79)\n"
        "    Validation: 2022-03-22 00:00:00 -- 2022-03-31 00:00:00  (n=10)\n"
        "Fold: 2\n"
        "    Training:   2022-01-02 00:00:00 -- 2022-03-31 00:00:00  (n=89)\n"
        "    Validation: 2022-04-01 00:00:00 -- 2022-04-10 00:00:00  (n=10)\n\n"
    )

    assert out == expected_out
    assert folds == expected


@pytest.mark.parametrize("skip_folds",
                         [3, [1, 2, 4, 5, 7]],
                         ids = lambda skip_folds: f'{skip_folds}')
@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70), range(68, 70), range(70, 80), range(70, 80), True],
                                  [range(30, 100), range(98, 100), range(100, 110), range(100, 110), True],
                                  [range(60, 130), range(128, 130), range(130, 140), range(130, 140), True]]),
                          (False, [[[0, 70], [68, 70], [70, 80], [70, 80], True],
                                   [[30, 100], [98, 100], [100, 110], [100, 110], True],
                                   [[60, 130], [128, 130], [130, 140], [130, 140], True]])],
                         ids = lambda argument: f'{argument}')
def test_TimeSeriesFold_split_refit_fixed_no_gap_no_remainder_skip_folds_3(capfd, skip_folds, return_all_indexes, expected):
    """
    Test TimeSeriesFold split method output when refit is True, fixed_train_size is 
    True, gap=0, not remainder and skip_folds=3.
    """
    y = pd.Series(np.arange(150))
    y.index = pd.date_range(start='2022-01-01', periods=150, freq='D')
    cv = TimeSeriesFold(
            steps                 = 10,
            initial_train_size    = 70,
            window_size           = 2,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = skip_folds,
            allow_incomplete_fold = True,
            return_all_indexes    = return_all_indexes,
        )
    folds = cv.split(X=y)

    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations used for initial training: 70\n"
        "Number of observations used for backtesting: 80\n"
        "    Number of folds: 8\n"
        "    Number skipped folds: 5 [1, 2, 4, 5, 7]\n"
        "    Number of steps per fold: 10\n"
        "    Number of steps to exclude from the end of each train set before test (gap): 0\n\n"
        "Fold: 0\n"
        "    Training:   2022-01-01 00:00:00 -- 2022-03-11 00:00:00  (n=70)\n"
        "    Validation: 2022-03-12 00:00:00 -- 2022-03-21 00:00:00  (n=10)\n"
        "Fold: 1\n"
        "    Fold skipped\n"
        "Fold: 2\n"
        "    Fold skipped\n"
        "Fold: 3\n"
        "    Training:   2022-01-31 00:00:00 -- 2022-04-10 00:00:00  (n=70)\n"
        "    Validation: 2022-04-11 00:00:00 -- 2022-04-20 00:00:00  (n=10)\n"
        "Fold: 4\n"
        "    Fold skipped\n"
        "Fold: 5\n"
        "    Fold skipped\n"
        "Fold: 6\n"
        "    Training:   2022-03-02 00:00:00 -- 2022-05-10 00:00:00  (n=70)\n"
        "    Validation: 2022-05-11 00:00:00 -- 2022-05-20 00:00:00  (n=10)\n"
        "Fold: 7\n"
        "    Fold skipped\n\n"
    )

    assert out == expected_out
    assert folds == expected


def test_timeseriesfold_split_raise_error_when_X_is_not_series_dataframe_or_dict():
    """
    Test that ValueError is raised when X is not a pd.Series, pd.DataFrame or dict.
    """
    X = np.arange(100)
    cv = TimeSeriesFold(steps=10, initial_train_size=70)
    msg = (
        f"X must be a pandas Series, DataFrame, Index or a dictionary. Got {type(X)}.")
    with pytest.raises(TypeError, match=msg):
        cv.split(X=X)