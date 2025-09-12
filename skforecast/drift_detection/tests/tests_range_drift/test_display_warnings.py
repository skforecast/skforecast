# Unit test _display_warnings
# ==============================================================================
import pytest
from skforecast.drift_detection import RangeDriftDetector
from skforecast.exceptions import FeatureOutOfRangeWarning

def test_display_warnings_numeric_without_series_name():
    not_compliant_feature = "feature1"
    feature_range = (1.0, 10.0)
    series_name = None

    expected_msg = (
        f"'{not_compliant_feature}' has values outside the range seen during training "
        f"[{feature_range[0]:.5f}, {feature_range[1]:.5f}]. "
        f"This may affect the accuracy of the predictions."
        f"\nYou can suppress this warning using: warnings.simplefilter('ignore', category=FeatureOutOfRangeWarning)"
    )
    # Capture warnings instead of using regex
    with pytest.warns(FeatureOutOfRangeWarning) as record:
        RangeDriftDetector._display_warnings(
            not_compliant_feature=not_compliant_feature,
            feature_range=feature_range,
            series_name=series_name
        )

    assert str(record[0].message) == expected_msg


def test_display_warnings_numeric_with_series_name():
    """
    Test _display_warnings with numeric feature_range and series_name.
    """
    not_compliant_feature = "feature1"
    feature_range = (1.0, 10.0)
    series_name = "series1"

    expected_msg = (
        f"'{series_name}': '{not_compliant_feature}' has values outside the range seen during training "
        f"[{feature_range[0]:.5f}, {feature_range[1]:.5f}]. "
        f"This may affect the accuracy of the predictions."
        f"\nYou can suppress this warning using: warnings.simplefilter('ignore', category=FeatureOutOfRangeWarning)"
    )
    with pytest.warns(FeatureOutOfRangeWarning) as record:
        RangeDriftDetector._display_warnings(
            not_compliant_feature=not_compliant_feature,
            feature_range=feature_range,
            series_name=series_name
        )
    assert str(record[0].message) == expected_msg


def test_display_warnings_categorical_without_series_name():
    """
    Test _display_warnings with categorical feature_range and no series_name.
    """
    not_compliant_feature = "feature1"
    feature_range = {"a", "b", "c"}
    series_name = None

    expected_msg = (
        f"'{not_compliant_feature}' has values not seen during training. Seen values: "
        f"{feature_range}. This may affect the accuracy of the predictions."
        f"\nYou can suppress this warning using: warnings.simplefilter('ignore', category=FeatureOutOfRangeWarning)"
    )
    with pytest.warns(FeatureOutOfRangeWarning) as record:
        RangeDriftDetector._display_warnings(
            not_compliant_feature=not_compliant_feature,
            feature_range=feature_range,
            series_name=series_name
        )
    assert str(record[0].message) == expected_msg


def test_display_warnings_categorical_with_series_name():
    """
    Test _display_warnings with categorical feature_range and series_name.
    """
    not_compliant_feature = "feature1"
    feature_range = {"a", "b", "c"}
    series_name = "series1"

    expected_msg = (
        f"'{series_name}': '{not_compliant_feature}' has values not seen during training. Seen values: "
        f"{feature_range}. This may affect the accuracy of the predictions."
        f"\nYou can suppress this warning using: warnings.simplefilter('ignore', category=FeatureOutOfRangeWarning)"
    )
    with pytest.warns(FeatureOutOfRangeWarning) as record:
        RangeDriftDetector._display_warnings(
            not_compliant_feature=not_compliant_feature,
            feature_range=feature_range,
            series_name=series_name
        )
    assert str(record[0].message) == expected_msg
