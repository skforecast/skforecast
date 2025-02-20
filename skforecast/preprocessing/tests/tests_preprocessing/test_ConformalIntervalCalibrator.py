# Unit test ConformalIntervalCalibrator
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.preprocessing import ConformalIntervalCalibrator



def test_TimeSeriesDifferentiator_validate_params():
    """
    ConformalIntervalCalibrator validate params.
    """

    err_msg = re.escape(
        "`nominal_coverage` must be a float between 0 and 1. Got 10"
    ) 
    with pytest.raises(ValueError, match = err_msg):
        ConformalIntervalCalibrator(order = 1000)


def test_fit_raise_error_invalid_y_true_type():

    y_true = 'invalid_type'
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5]
    })
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    msg = "`y_true` must be a pandas Series, pandas DataFrame, or a dictionary."
    with pytest.raises(ValueError, match=msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)

def test_fit_raises_error_invalid_y_pred_interval_type():

    y_true = pd.Series([1, 2, 3, 4, 5], name='y')
    y_pred_interval = 'invalid_type'
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    msg="`y_pred_interval` must be a pandas DataFrame."
    with pytest.raises(ValueError, match=msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)

def test_fit_missing_columns_in_y_pred_interval():

    y_true = pd.Series([1, 2, 3, 4, 5], name='y')
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5]
    })
    msg="`y_pred_interval` must have columns 'lower_bound' and 'upper_bound'."
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    with pytest.raises(ValueError, match=msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)

def test_fit_missing_level_in_y_pred_interval_when_y_true_is_dataframe_or_dict():

    y_true = pd.DataFrame({
        'y1': [1, 2, 3, 4, 5],
        'y2': [1, 2, 3, 4, 5]
    })
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5]
    })
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    msg = (
        "If `y_true` is a pandas DataFrame or a dictionary, `y_pred_interval` "
        "must have an additional column 'level' to identify each series."
    )
    with pytest.raises(ValueError, match=msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)

    y_true = {
        'y1': pd.Series([1, 2, 3, 4, 5], name='y1'),
        'y2': pd.Series([1, 2, 3, 4, 5], name='y2')
    }
    with pytest.raises(ValueError, match=msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)

def test_fit_when_y_true_is_series_without_name():

    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5]
    })
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)

    assert calibrator.fit_series_names_ == ['y']
    assert calibrator.fit_input_type_ == "single_series"
    assert calibrator.correction_factor_ == {'y': -0.5}


def test_fit_raise_error_when_y_true_is_single_series_and_y_pred_interval_is_dataframe_with_multiple_levels():

    y_true = pd.Series([1, 2, 3, 4, 5], name='series_1')
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5],
        'level': ['series_1', 'series_1', 'series_1', 'series_2', 'series_2']
    })
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    msg = (
        "If `y_true` is a pandas Series, `y_pred_interval` must have "
        "only one series. Found multiple values in column 'level'."
    )
    with pytest.raises(ValueError, match=msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)

def test_fit_raise_error_when_y_true_is_not_in_y_true_interval():

    y_true = pd.Series([1, 2, 3, 4, 5], name='series_1')
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5],
        'level': ['series_2', 'series_2', 'series_2', 'series_2', 'series_2']
    })
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    msg = re.escape(
        f"Series name in `y_true` ({y_true.name}) does not match the level "
        f"name in `y_pred_interval` ({y_pred_interval['level'].unique()[0]})."
    )
    with pytest.raises(ValueError, match=msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)


def test_fit_raise_error_when_series_names_not_match_in_y_true_and_y_pred_interval():

    y_true = pd.DataFrame({
        'y1': [1, 2, 3, 4, 5],
        'y2': [1, 2, 3, 4, 5]
    })
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5],
        'level': ['series_1', 'series_1', 'series_2', 'series_2', 'series_2']
    })
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    msg = "Series names in `y_true` and `y_pred_interval` do not match."
    
    with pytest.raises(ValueError, match=msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)


def test_fit_raise_error_when_y_true_is_dict_with_invalid_types_as_values():

    y_true = {
        'series_1': [1, 2, 3, 4, 5],
        'series_2': [1, 2, 3, 4, 5]
    }
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5],
        'level': ['series_1', 'series_1', 'series_2', 'series_2', 'series_2']
    })
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    msg = "All values in `y_true` must be pandas Series. Got <class 'list'>."
    
    with pytest.raises(ValueError, match=msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)

def test_fit_raise_error_when_indexex_in_y_true_and_y_pred_interval_do_not_match():

    y_true = pd.Series([1, 2, 3, 4, 5], name='series_1', index=[1, 2, 3, 4, 5])
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5],
        'level': ['series_1', 'series_1', 'series_1', 'series_1', 'series_1']
    }, index=[10, 20, 30, 40, 50])
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    msg = 'Index of `y_true` and `y_pred_interval` must match.'
    
    with pytest.raises(ValueError, match=msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)


def test_fit_and_transform_for_single_series():

    # Simulate intervals and y_true for a single series
    rng = np.random.default_rng(42)
    prediction_interval = pd.DataFrame({
            'lower_bound': np.sin(np.linspace(0, 4 * np.pi, 100)),
            'upper_bound': np.sin(np.linspace(0, 4 * np.pi, 100)) + 5
        },
        index=pd.date_range(start='2024-01-01', periods=100, freq='D')
    )
    y_true = (prediction_interval['lower_bound'] + prediction_interval['upper_bound']) / 2
    y_true.name = "series_1"
    y_true.iloc[1::5] = prediction_interval.iloc[1::5, 0] - rng.normal(1, 1, 20)
    y_true.iloc[3::5] = prediction_interval.iloc[1::5, 1] + rng.normal(1, 1, 20)
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    calibrator.fit(y_true=y_true, y_pred_interval=prediction_interval)
    results = calibrator.transform(prediction_interval)

    expected_results = pd.DataFrame({
        'level': 'series_1',
        'lower_bound': np.array(
                        [-1.03424353, -0.90765108, -0.78309555, -0.66258108, -0.5480468 ,
                        -0.4413356 , -0.34416452, -0.25809707, -0.1845181 , -0.12461154,
                        -0.07934129, -0.04943578, -0.03537619, -0.03738876, -0.05544109,
                        -0.08924271, -0.13824976, -0.20167368, -0.27849396, -0.36747453,
                        -0.46718367, -0.57601701, -0.69222339, -0.813933  , -0.93918749,
                        -1.06597147, -1.19224493, -1.31597609, -1.43517407, -1.54792092,
                        -1.65240252, -1.7469377 , -1.83000537, -1.90026894, -1.95659783,
                        -1.99808569, -2.02406498, -2.03411766, -2.028082  , -2.0060551 ,
                        -1.96839139, -1.9156969 , -1.84881949, -1.76883524, -1.67703114,
                        -1.57488435, -1.46403845, -1.34627698, -1.22349478, -1.09766745,
                        -0.97081961, -0.84499229, -0.72221009, -0.60444862, -0.49360272,
                        -0.39145592, -0.29965182, -0.21966758, -0.15279017, -0.10009567,
                        -0.06243197, -0.04040507, -0.03436941, -0.04442209, -0.07040137,
                        -0.11188924, -0.16821813, -0.23848169, -0.32154936, -0.41608455,
                        -0.52056614, -0.633313  , -0.75251098, -0.87624214, -1.0025156 ,
                        -1.12929958, -1.25455407, -1.37626368, -1.49247006, -1.6013034 ,
                        -1.70101253, -1.78999311, -1.86681339, -1.93023731, -1.97924435,
                        -2.01304598, -2.03109831, -2.03311087, -2.01905129, -1.98914577,
                        -1.94387553, -1.88396896, -1.81039   , -1.72432254, -1.62715146,
                        -1.52044027, -1.40590599, -1.28539152, -1.16083599, -1.03424353]
                       ),
        'upper_bound': np.array([
                        6.03424353, 6.16083599, 6.28539152, 6.40590599, 6.52044027,
                        6.62715146, 6.72432254, 6.81039   , 6.88396896, 6.94387553,
                        6.98914577, 7.01905129, 7.03311087, 7.03109831, 7.01304598,
                        6.97924435, 6.93023731, 6.86681339, 6.78999311, 6.70101253,
                        6.6013034 , 6.49247006, 6.37626368, 6.25455407, 6.12929958,
                        6.0025156 , 5.87624214, 5.75251098, 5.633313  , 5.52056614,
                        5.41608455, 5.32154936, 5.23848169, 5.16821813, 5.11188924,
                        5.07040137, 5.04442209, 5.03436941, 5.04040507, 5.06243197,
                        5.10009567, 5.15279017, 5.21966758, 5.29965182, 5.39145592,
                        5.49360272, 5.60444862, 5.72221009, 5.84499229, 5.97081961,
                        6.09766745, 6.22349478, 6.34627698, 6.46403845, 6.57488435,
                        6.67703114, 6.76883524, 6.84881949, 6.9156969 , 6.96839139,
                        7.0060551 , 7.028082  , 7.03411766, 7.02406498, 6.99808569,
                        6.95659783, 6.90026894, 6.83000537, 6.7469377 , 6.65240252,
                        6.54792092, 6.43517407, 6.31597609, 6.19224493, 6.06597147,
                        5.93918749, 5.813933  , 5.69222339, 5.57601701, 5.46718367,
                        5.36747453, 5.27849396, 5.20167368, 5.13824976, 5.08924271,
                        5.05544109, 5.03738876, 5.03537619, 5.04943578, 5.07934129,
                        5.12461154, 5.1845181 , 5.25809707, 5.34416452, 5.4413356 ,
                        5.5480468 , 5.66258108, 5.78309555, 5.90765108, 6.03424353]
                    ,)
    },
    index=pd.date_range(start='2024-01-01', periods=100, freq='D')
    )

    assert calibrator.nominal_coverage == 0.8
    assert calibrator.correction_factor_ == {'series_1': 1.0342435333380045}
    assert calibrator.fit_series_names_ == ['series_1']
    pd.testing.assert_frame_equal(results, expected_results)


def test_fit_and_transform_for_multi_series():

    # Simulate intervals and y_true for a single series
    rng = np.random.default_rng(42)
    prediction_interval = pd.DataFrame({
            'lower_bound': np.sin(np.linspace(0, 4 * np.pi, 100)),
            'upper_bound': np.sin(np.linspace(0, 4 * np.pi, 100)) + 5
        },
        index=pd.date_range(start='2024-01-01', periods=100, freq='D')
    )
    y_true = (prediction_interval['lower_bound'] + prediction_interval['upper_bound']) / 2
    y_true.iloc[1::5] = prediction_interval.iloc[1::5, 0] - rng.normal(1, 1, 20)
    y_true.iloc[3::5] = prediction_interval.iloc[1::5, 1] + rng.normal(1, 1, 20)

    # Simulate intervals and y_true for three series repeating the same values
    y_true_multiseries = pd.DataFrame({
                            'series_1': y_true,
                            'series_2': y_true,
                            'series_3': y_true
                        })
    prediction_interval_multiseries = pd.concat([prediction_interval]*3, axis=0)
    prediction_interval_multiseries['level'] = np.repeat(['series_1', 'series_2', 'series_3'], 100)

    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    calibrator.fit(y_true=y_true_multiseries, y_pred_interval=prediction_interval_multiseries)
    results = calibrator.transform(prediction_interval_multiseries)

    expected_results = pd.DataFrame({
        'level': 'series_1',
        'lower_bound': np.array(
                        [-1.03424353, -0.90765108, -0.78309555, -0.66258108, -0.5480468 ,
                        -0.4413356 , -0.34416452, -0.25809707, -0.1845181 , -0.12461154,
                        -0.07934129, -0.04943578, -0.03537619, -0.03738876, -0.05544109,
                        -0.08924271, -0.13824976, -0.20167368, -0.27849396, -0.36747453,
                        -0.46718367, -0.57601701, -0.69222339, -0.813933  , -0.93918749,
                        -1.06597147, -1.19224493, -1.31597609, -1.43517407, -1.54792092,
                        -1.65240252, -1.7469377 , -1.83000537, -1.90026894, -1.95659783,
                        -1.99808569, -2.02406498, -2.03411766, -2.028082  , -2.0060551 ,
                        -1.96839139, -1.9156969 , -1.84881949, -1.76883524, -1.67703114,
                        -1.57488435, -1.46403845, -1.34627698, -1.22349478, -1.09766745,
                        -0.97081961, -0.84499229, -0.72221009, -0.60444862, -0.49360272,
                        -0.39145592, -0.29965182, -0.21966758, -0.15279017, -0.10009567,
                        -0.06243197, -0.04040507, -0.03436941, -0.04442209, -0.07040137,
                        -0.11188924, -0.16821813, -0.23848169, -0.32154936, -0.41608455,
                        -0.52056614, -0.633313  , -0.75251098, -0.87624214, -1.0025156 ,
                        -1.12929958, -1.25455407, -1.37626368, -1.49247006, -1.6013034 ,
                        -1.70101253, -1.78999311, -1.86681339, -1.93023731, -1.97924435,
                        -2.01304598, -2.03109831, -2.03311087, -2.01905129, -1.98914577,
                        -1.94387553, -1.88396896, -1.81039   , -1.72432254, -1.62715146,
                        -1.52044027, -1.40590599, -1.28539152, -1.16083599, -1.03424353]
                       ),
        'upper_bound': np.array([
                        6.03424353, 6.16083599, 6.28539152, 6.40590599, 6.52044027,
                        6.62715146, 6.72432254, 6.81039   , 6.88396896, 6.94387553,
                        6.98914577, 7.01905129, 7.03311087, 7.03109831, 7.01304598,
                        6.97924435, 6.93023731, 6.86681339, 6.78999311, 6.70101253,
                        6.6013034 , 6.49247006, 6.37626368, 6.25455407, 6.12929958,
                        6.0025156 , 5.87624214, 5.75251098, 5.633313  , 5.52056614,
                        5.41608455, 5.32154936, 5.23848169, 5.16821813, 5.11188924,
                        5.07040137, 5.04442209, 5.03436941, 5.04040507, 5.06243197,
                        5.10009567, 5.15279017, 5.21966758, 5.29965182, 5.39145592,
                        5.49360272, 5.60444862, 5.72221009, 5.84499229, 5.97081961,
                        6.09766745, 6.22349478, 6.34627698, 6.46403845, 6.57488435,
                        6.67703114, 6.76883524, 6.84881949, 6.9156969 , 6.96839139,
                        7.0060551 , 7.028082  , 7.03411766, 7.02406498, 6.99808569,
                        6.95659783, 6.90026894, 6.83000537, 6.7469377 , 6.65240252,
                        6.54792092, 6.43517407, 6.31597609, 6.19224493, 6.06597147,
                        5.93918749, 5.813933  , 5.69222339, 5.57601701, 5.46718367,
                        5.36747453, 5.27849396, 5.20167368, 5.13824976, 5.08924271,
                        5.05544109, 5.03738876, 5.03537619, 5.04943578, 5.07934129,
                        5.12461154, 5.1845181 , 5.25809707, 5.34416452, 5.4413356 ,
                        5.5480468 , 5.66258108, 5.78309555, 5.90765108, 6.03424353]
                    ,)
    },
    index=pd.date_range(start='2024-01-01', periods=100, freq='D')
    )

    expected_results = pd.concat([expected_results] * 3, axis=0)
    expected_results['level'] = np.repeat(['series_1', 'series_2', 'series_3'], 100)

    assert calibrator.nominal_coverage == 0.8
    assert calibrator.correction_factor_ == {'series_1': 1.0342435333380045, 'series_2': 1.0342435333380045, 'series_3': 1.0342435333380045}
    assert calibrator.fit_series_names_ == ['series_1', 'series_2', 'series_3'] 
    pd.testing.assert_frame_equal(results, expected_results)