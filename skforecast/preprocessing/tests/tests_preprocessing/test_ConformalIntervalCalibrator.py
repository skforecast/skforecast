# Unit test ConformalIntervalCalibrator
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.preprocessing import ConformalIntervalCalibrator


def test_init_validate_params():
    """
    ConformalIntervalCalibrator validate params.
    """
    err_msg = re.escape(
        "`nominal_coverage` must be a float between 0 and 1. Got 90"
    ) 
    with pytest.raises(ValueError, match = err_msg):
        ConformalIntervalCalibrator(nominal_coverage=90)


def test_fit_validate_params():
    """
    Test fit check params.
    """
    y_true = pd.Series([1, 2, 3, 4, 5], name='y')
    y_true_df = pd.DataFrame({'y1': [1, 2, 3, 4, 5], 'y2': [1, 2, 3, 4, 5]})
    y_true_dict = {
        'y1': pd.Series([1, 2, 3, 4, 5], name='y1'),
        'y2': pd.Series([1, 2, 3, 4, 5], name='y2')
    }
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5]
    })
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)

    y_true_invalid = 'invalid_type'
    err_msg = re.escape(
        "`y_true` must be a pandas Series, pandas DataFrame, or a dictionary."
    )
    with pytest.raises(TypeError, match=err_msg):
        calibrator.fit(y_true=y_true_invalid, y_pred_interval=y_pred_interval)

    y_pred_invalid = 'invalid_type'
    err_msg = re.escape(
        "`y_pred_interval` must be a pandas DataFrame."
    )
    with pytest.raises(TypeError, match=err_msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_invalid)
    
    y_pred_invalid = pd.DataFrame({'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5]})
    err_msg = re.escape(
        "`y_pred_interval` must have columns 'lower_bound' and 'upper_bound'."
    )
    with pytest.raises(ValueError, match=err_msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_invalid)

    y_pred_invalid = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5]
    })
    err_msg = re.escape(
        "If `y_true` is a pandas DataFrame or a dictionary, `y_pred_interval` "
        "must have an additional column 'level' to identify each series."
    )
    with pytest.raises(ValueError, match=err_msg):
        calibrator.fit(y_true=y_true_df, y_pred_interval=y_pred_invalid)
    with pytest.raises(ValueError, match=err_msg):
        calibrator.fit(y_true=y_true_dict, y_pred_interval=y_pred_invalid)

    y_pred_invalid = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5],
        'level': ['y', 'y', 'y', 'y_2', 'y_2']
    })
    err_msg = re.escape(
        "If `y_true` is a pandas Series, `y_pred_interval` must have "
        "only one series. Found multiple values in column 'level'."
    )
    with pytest.raises(ValueError, match=err_msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_invalid)

    y_pred_invalid = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5],
        'level': ['not_y', 'not_y', 'not_y', 'not_y', 'not_y']
    })
    err_msg = re.escape(
        "Series name in `y_true`, 'y', does not match the level "
        "name in `y_pred_interval`, 'not_y'."
    )
    with pytest.raises(ValueError, match=err_msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_invalid)

    y_true_invalid = {
        'y1': pd.Series([1, 2, 3, 4, 5], name='y1').to_frame(),
        'y2': pd.Series([1, 2, 3, 4, 5], name='y2')
    }
    y_pred_interval = pd.DataFrame({
            'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5, 0.5, 1.5, 2.5, 3.5, 4.5],
            'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            'level': ['series_3'] * 5 + ['series_4'] * 5
        }, 
        index=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    )
    err_msg = re.escape(
        f"When `y_true` is a dict, all its values must be pandas "
        f"Series. Got {type(y_true_invalid['y1'])} for series 'y1'."
    )
    with pytest.raises(ValueError, match=err_msg):
        calibrator.fit(y_true=y_true_invalid, y_pred_interval=y_pred_interval)
    
    y_true = pd.DataFrame({
            'series_1': [1, 2, 3, 4, 5],
            'series_2': [1, 2, 3, 4, 5]
        }, 
        index=[0, 1, 2, 3, 4]
    )
    y_pred_interval = pd.DataFrame({
            'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5, 0.5, 1.5, 2.5, 3.5, 4.5],
            'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            'level': ['series_3'] * 5 + ['series_4'] * 5
        }, 
        index=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    )
    err_msg = re.escape(
        "Series names in `y_true` and `y_pred_interval` do not match.\n"
        "   `y_true` series names          : ['series_1', 'series_2']\n"
        "   `y_pred_interval` series names : ['series_3', 'series_4']"
    )
    with pytest.raises(ValueError, match=err_msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)

    y_true = pd.Series([1, 2, 3, 4, 5], name='series_1', index=[1, 2, 3, 4, 5])
    y_pred_interval = pd.DataFrame({
            'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
            'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5],
            'level': ['series_1', 'series_1', 'series_1', 'series_1', 'series_1']
        }, 
        index=[10, 20, 30, 40, 50]
    )
    err_msg = re.escape(
        "Index of `y_true` and `y_pred_interval` must match. Different "
        "indices found for series 'series_1'."
    )
    with pytest.raises(IndexError, match=err_msg):
        calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)


def test_fit_when_y_true_is_series_without_name():
    """
    Test fit when y_true is a pandas Series without name.
    """
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5]
    })
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)

    assert calibrator.is_fitted is True
    assert calibrator.fit_series_names_ == ['y']
    assert calibrator.fit_input_type_ == "single_series"
    assert calibrator.correction_factor_ == {'y': -0.5}


@pytest.mark.parametrize("y_true", 
    [pd.DataFrame({'series_1': [1, 2, 3, 4, 5], 'series_2': [10, 20, 30, 40, 50]}),
     {'series_1': pd.Series([1, 2, 3, 4, 5]), 'series_2': pd.Series([10, 20, 30, 40, 50])}], 
    ids = lambda y_true: f'y_true: {type(y_true)}')
def test_fit_when_y_true_is_pandas_DataFrame_or_dict_with_two_series(y_true):
    """
    Test fit when y_true is a pandas DataFrame or dict with two series.
    """
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5, 5, 15, 25, 35, 45],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5, 15, 25, 35, 45, 55],
        'level': ['series_1'] * 5 + ['series_2'] * 5},
        index = pd.Index([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    )
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)
    calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)

    assert calibrator.is_fitted is True
    assert calibrator.fit_series_names_ == ['series_1', 'series_2']
    assert calibrator.fit_input_type_ == "multiple_series"
    assert calibrator.correction_factor_ == {'series_1': -0.5, 'series_2': -5}


def test_transform_validate_params():
    """
    Test transform check params.
    """
    y_true = pd.Series([1, 2, 3, 4, 5], name='y')
    y_true_df = pd.DataFrame({'series_1': [1, 2, 3, 4, 5], 'series_2': [1, 2, 3, 4, 5]})
    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5]
    })
    y_pred_interval_multiple = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        'level': ['series_1'] * 5 + ['series_2'] * 5
    }, index=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8)

    err_msg = re.escape(
        "ConformalIntervalCalibrator not fitted yet. Call 'fit' with "
        "training data first."
    )
    with pytest.raises(NotFittedError, match=err_msg):
        calibrator.transform(y_pred_interval=y_pred_interval)
    
    calibrator.fit(y_true=y_true, y_pred_interval=y_pred_interval)
    
    err_msg = re.escape("`y_pred_interval` must be a pandas DataFrame.")
    with pytest.raises(TypeError, match=err_msg):
        calibrator.transform(y_pred_interval='invalid_type')
    
    y_pred_invalid = pd.DataFrame({
        'col_1': [0.5, 1.5, 2.5, 3.5, 4.5],
        'col_2': [1.5, 2.5, 3.5, 4.5, 5.5],
    })
    err_msg = re.escape(
        "`y_pred_interval` must have columns 'lower_bound' and 'upper_bound'."
    )
    with pytest.raises(ValueError, match=err_msg):
        calibrator.transform(y_pred_interval=y_pred_invalid)
    
    calibrator.fit(y_true=y_true_df, y_pred_interval=y_pred_interval_multiple)
    err_msg = re.escape(
        "The transformer was fitted with multiple series. `y_pred_interval` "
        "must be a long-format DataFrame with three columns: 'level', "
        "'lower_bound', and 'upper_bound'. The 'level' column identifies "
        "the series to which each interval belongs."
    )
    with pytest.raises(ValueError, match=err_msg):
        calibrator.transform(y_pred_interval=y_pred_interval)

    y_pred_interval = pd.DataFrame({
        'lower_bound': [0.5, 1.5, 2.5, 3.5, 4.5,],
        'upper_bound': [1.5, 2.5, 3.5, 4.5, 5.5],
        'level': ['series_3'] * 5
    })
    err_msg = re.escape(
        "Series 'series_3' was not seen during fit. Available series are: "
        "['series_1', 'series_2']."
    )
    with pytest.raises(ValueError, match=err_msg):
        calibrator.transform(y_pred_interval=y_pred_interval)


def test_fit_and_transform_for_single_series_symmetric():
    """
    Test fit and transform for a single series with symmetric calibration.
    """
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

    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8, symmetric_calibration=True)
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
    assert calibrator.correction_factor_lower_ == {'series_1': 1.072211668121823}
    assert calibrator.correction_factor_upper_ == {'series_1': 0.9897729203096686}
    assert calibrator.fit_series_names_ == ['series_1']
    pd.testing.assert_frame_equal(results, expected_results)


def test_fit_and_transform_for_single_series_symmetric_False():
    """
    Test fit and transform for a single series with no symmetric calibration.
    """
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

    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8, symmetric_calibration=False)
    calibrator.fit(y_true=y_true, y_pred_interval=prediction_interval)
    results = calibrator.transform(prediction_interval)

    expected_results = pd.DataFrame({
        'level': 'series_1',
        'lower_bound': np.array(
                        [-1.07221167, -0.94561921, -0.82106368, -0.70054921, -0.58601493,
                        -0.47930374, -0.38213266, -0.2960652 , -0.22248624, -0.16257967,
                        -0.11730943, -0.08740392, -0.07334433, -0.07535689, -0.09340922,
                        -0.12721085, -0.17621789, -0.23964181, -0.31646209, -0.40544267,
                        -0.5051518 , -0.61398515, -0.73019152, -0.85190114, -0.97715562,
                        -1.1039396 , -1.23021306, -1.35394422, -1.4731422 , -1.58588906,
                        -1.69037065, -1.78490584, -1.86797351, -1.93823707, -1.99456596,
                        -2.03605383, -2.06203311, -2.0720858 , -2.06605013, -2.04402324,
                        -2.00635953, -1.95366503, -1.88678762, -1.80680338, -1.71499928,
                        -1.61285249, -1.50200658, -1.38424511, -1.26146291, -1.13563559,
                        -1.00878775, -0.88296042, -0.76017822, -0.64241676, -0.53157085,
                        -0.42942406, -0.33761996, -0.25763572, -0.1907583 , -0.13806381,
                        -0.1004001 , -0.0783732 , -0.07233754, -0.08239023, -0.10836951,
                        -0.14985737, -0.20618626, -0.27644983, -0.3595175 , -0.45405268,
                        -0.55853428, -0.67128113, -0.79047911, -0.91421027, -1.04048373,
                        -1.16726771, -1.2925222 , -1.41423181, -1.53043819, -1.63927153,
                        -1.73898067, -1.82796124, -1.90478152, -1.96820544, -2.01721249,
                        -2.05101411, -2.06906644, -2.07107901, -2.05701942, -2.02711391,
                        -1.98184366, -1.9219371 , -1.84835813, -1.76229068, -1.6651196 ,
                        -1.5584084 , -1.44387412, -1.32335966, -1.19880412, -1.07221167]
                        ),
        'upper_bound': np.array([
                        5.98977292, 6.11636537, 6.24092091, 6.36143538, 6.47596966,
                        6.58268085, 6.67985193, 6.76591938, 6.83949835, 6.89940492,
                        6.94467516, 6.97458067, 6.98864026, 6.9866277 , 6.96857537,
                        6.93477374, 6.88576669, 6.82234277, 6.74552249, 6.65654192,
                        6.55683278, 6.44799944, 6.33179306, 6.21008345, 6.08482896,
                        5.95804499, 5.83177152, 5.70804036, 5.58884238, 5.47609553,
                        5.37161393, 5.27707875, 5.19401108, 5.12374752, 5.06741863,
                        5.02593076, 4.99995148, 4.98989879, 4.99593446, 5.01796135,
                        5.05562506, 5.10831956, 5.17519697, 5.25518121, 5.34698531,
                        5.4491321 , 5.55997801, 5.67773947, 5.80052168, 5.926349  ,
                        6.05319684, 6.17902416, 6.30180637, 6.41956783, 6.53041374,
                        6.63256053, 6.72436463, 6.80434887, 6.87122628, 6.92392078,
                        6.96158449, 6.98361138, 6.98964705, 6.97959436, 6.95361508,
                        6.91212721, 6.85579832, 6.78553476, 6.70246709, 6.60793191,
                        6.50345031, 6.39070346, 6.27150548, 6.14777432, 6.02150085,
                        5.89471688, 5.76946239, 5.64775278, 5.5315464 , 5.42271306,
                        5.32300392, 5.23402335, 5.15720307, 5.09377915, 5.0447721 ,
                        5.01097047, 4.99291814, 4.99090558, 5.00496517, 5.03487068,
                        5.08014092, 5.14004749, 5.21362646, 5.29969391, 5.39686499,
                        5.50357618, 5.61811046, 5.73862493, 5.86318047, 5.98977292]
                    )
    },
    index=pd.date_range(start='2024-01-01', periods=100, freq='D')
    )

    assert calibrator.nominal_coverage == 0.8
    assert calibrator.correction_factor_ == {'series_1': 1.0342435333380045}
    assert calibrator.correction_factor_lower_ == {'series_1': 1.072211668121823}
    assert calibrator.correction_factor_upper_ == {'series_1': 0.9897729203096686}
    assert calibrator.fit_series_names_ == ['series_1']
    pd.testing.assert_frame_equal(results, expected_results)


def test_fit_and_transform_for_multi_series_symmetric():
    """
    Test fit and transform for multiple series with symmetric calibration.
    """
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
    prediction_interval_multiseries = pd.concat([prediction_interval] * 3, axis=0)
    prediction_interval_multiseries['level'] = np.repeat(['series_1', 'series_2', 'series_3'], 100)

    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8, symmetric_calibration=True)
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
    assert calibrator.correction_factor_ == {
        "series_1": 1.0342435333380045,
        "series_2": 1.0342435333380045,
        "series_3": 1.0342435333380045,
    }
    assert calibrator.correction_factor_lower_ == {
        "series_1": 1.072211668121823,
        "series_2": 1.072211668121823,
        "series_3": 1.072211668121823,
    }
    assert calibrator.correction_factor_upper_ == {
        "series_1": 0.9897729203096686,
        "series_2": 0.9897729203096686,
        "series_3": 0.9897729203096686,
    }
    assert calibrator.fit_series_names_ == ["series_1", "series_2", "series_3"]
    pd.testing.assert_frame_equal(results, expected_results)


def test_fit_and_transform_for_multi_series_symmetric_False():
    """
    Test fit and transform for multiple series with no symmetric calibration.
    """
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
    prediction_interval_multiseries = pd.concat([prediction_interval] * 3, axis=0)
    prediction_interval_multiseries['level'] = np.repeat(['series_1', 'series_2', 'series_3'], 100)

    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.8, symmetric_calibration=False)
    calibrator.fit(y_true=y_true_multiseries, y_pred_interval=prediction_interval_multiseries)
    results = calibrator.transform(prediction_interval_multiseries)

    expected_results = pd.DataFrame({
        'level': 'series_1',
        'lower_bound': np.array(
                        [-1.07221167, -0.94561921, -0.82106368, -0.70054921, -0.58601493,
                        -0.47930374, -0.38213266, -0.2960652 , -0.22248624, -0.16257967,
                        -0.11730943, -0.08740392, -0.07334433, -0.07535689, -0.09340922,
                        -0.12721085, -0.17621789, -0.23964181, -0.31646209, -0.40544267,
                        -0.5051518 , -0.61398515, -0.73019152, -0.85190114, -0.97715562,
                        -1.1039396 , -1.23021306, -1.35394422, -1.4731422 , -1.58588906,
                        -1.69037065, -1.78490584, -1.86797351, -1.93823707, -1.99456596,
                        -2.03605383, -2.06203311, -2.0720858 , -2.06605013, -2.04402324,
                        -2.00635953, -1.95366503, -1.88678762, -1.80680338, -1.71499928,
                        -1.61285249, -1.50200658, -1.38424511, -1.26146291, -1.13563559,
                        -1.00878775, -0.88296042, -0.76017822, -0.64241676, -0.53157085,
                        -0.42942406, -0.33761996, -0.25763572, -0.1907583 , -0.13806381,
                        -0.1004001 , -0.0783732 , -0.07233754, -0.08239023, -0.10836951,
                        -0.14985737, -0.20618626, -0.27644983, -0.3595175 , -0.45405268,
                        -0.55853428, -0.67128113, -0.79047911, -0.91421027, -1.04048373,
                        -1.16726771, -1.2925222 , -1.41423181, -1.53043819, -1.63927153,
                        -1.73898067, -1.82796124, -1.90478152, -1.96820544, -2.01721249,
                        -2.05101411, -2.06906644, -2.07107901, -2.05701942, -2.02711391,
                        -1.98184366, -1.9219371 , -1.84835813, -1.76229068, -1.6651196 ,
                        -1.5584084 , -1.44387412, -1.32335966, -1.19880412, -1.07221167]
                        ),
        'upper_bound': np.array([
                        5.98977292, 6.11636537, 6.24092091, 6.36143538, 6.47596966,
                        6.58268085, 6.67985193, 6.76591938, 6.83949835, 6.89940492,
                        6.94467516, 6.97458067, 6.98864026, 6.9866277 , 6.96857537,
                        6.93477374, 6.88576669, 6.82234277, 6.74552249, 6.65654192,
                        6.55683278, 6.44799944, 6.33179306, 6.21008345, 6.08482896,
                        5.95804499, 5.83177152, 5.70804036, 5.58884238, 5.47609553,
                        5.37161393, 5.27707875, 5.19401108, 5.12374752, 5.06741863,
                        5.02593076, 4.99995148, 4.98989879, 4.99593446, 5.01796135,
                        5.05562506, 5.10831956, 5.17519697, 5.25518121, 5.34698531,
                        5.4491321 , 5.55997801, 5.67773947, 5.80052168, 5.926349  ,
                        6.05319684, 6.17902416, 6.30180637, 6.41956783, 6.53041374,
                        6.63256053, 6.72436463, 6.80434887, 6.87122628, 6.92392078,
                        6.96158449, 6.98361138, 6.98964705, 6.97959436, 6.95361508,
                        6.91212721, 6.85579832, 6.78553476, 6.70246709, 6.60793191,
                        6.50345031, 6.39070346, 6.27150548, 6.14777432, 6.02150085,
                        5.89471688, 5.76946239, 5.64775278, 5.5315464 , 5.42271306,
                        5.32300392, 5.23402335, 5.15720307, 5.09377915, 5.0447721 ,
                        5.01097047, 4.99291814, 4.99090558, 5.00496517, 5.03487068,
                        5.08014092, 5.14004749, 5.21362646, 5.29969391, 5.39686499,
                        5.50357618, 5.61811046, 5.73862493, 5.86318047, 5.98977292]
                    )
    },
    index=pd.date_range(start='2024-01-01', periods=100, freq='D')
    )

    expected_results = pd.concat([expected_results] * 3, axis=0)
    expected_results['level'] = np.repeat(['series_1', 'series_2', 'series_3'], 100)

    assert calibrator.nominal_coverage == 0.8
    assert calibrator.correction_factor_ == {
        "series_1": 1.0342435333380045,
        "series_2": 1.0342435333380045,
        "series_3": 1.0342435333380045,
    }
    assert calibrator.correction_factor_lower_ == {
        "series_1": 1.072211668121823,
        "series_2": 1.072211668121823,
        "series_3": 1.072211668121823,
    }
    assert calibrator.correction_factor_upper_ == {
        "series_1": 0.9897729203096686,
        "series_2": 0.9897729203096686,
        "series_3": 0.9897729203096686,
    }
    assert calibrator.fit_series_names_ == ["series_1", "series_2", "series_3"]
    pd.testing.assert_frame_equal(results, expected_results)
