# Unit test _summary
# ==============================================================================
import pytest
from skforecast.drift_detection import RangeDriftDetector


def test_summary_no_out_of_range(capsys):
    """
    Test _summary when no out-of-range values are found.
    """
    out_of_range_series = []
    out_of_range_series_ranges = []
    out_of_range_exog = []
    out_of_range_exog_ranges = []
    out_of_range_exog_series_id = []

    RangeDriftDetector._summary(
        out_of_range_series=out_of_range_series,
        out_of_range_series_ranges=out_of_range_series_ranges,
        out_of_range_exog=out_of_range_exog,
        out_of_range_exog_ranges=out_of_range_exog_ranges,
        out_of_range_exog_series_id=out_of_range_exog_series_id
    )

    captured = capsys.readouterr()
    assert "No series with out-of-range values found." in captured.out
    assert "No exogenous variables with out-of-range values found." in captured.out


def test_summary_out_of_range_series(capsys):
    """
    Test _summary with out-of-range series.
    """
    out_of_range_series = ["series1", "series2"]
    out_of_range_series_ranges = [(1.0, 10.0), (2.0, 20.0)]
    out_of_range_exog = []
    out_of_range_exog_ranges = []
    out_of_range_exog_series_id = []

    RangeDriftDetector._summary(
        out_of_range_series=out_of_range_series,
        out_of_range_series_ranges=out_of_range_series_ranges,
        out_of_range_exog=out_of_range_exog,
        out_of_range_exog_ranges=out_of_range_exog_ranges,
        out_of_range_exog_series_id=out_of_range_exog_series_id
    )

    captured = capsys.readouterr()
    assert "'series1' has values outside the observed range [1.00000, 10.00000]." in captured.out
    assert "'series2' has values outside the observed range [2.00000, 20.00000]." in captured.out
    assert "No exogenous variables with out-of-range values found." in captured.out


def test_summary_out_of_range_exog(capsys):
    """
    Test _summary with out-of-range exogenous variables.
    """
    out_of_range_series = []
    out_of_range_series_ranges = []
    out_of_range_exog = ["exog1", "exog2"]
    out_of_range_exog_ranges = [(0.0, 5.0), (10.0, 15.0)]
    out_of_range_exog_series_id = ["series_1", "series_1"]

    RangeDriftDetector._summary(
        out_of_range_series=out_of_range_series,
        out_of_range_series_ranges=out_of_range_series_ranges,
        out_of_range_exog=out_of_range_exog,
        out_of_range_exog_ranges=out_of_range_exog_ranges,
        out_of_range_exog_series_id=out_of_range_exog_series_id
    )

    captured = capsys.readouterr()
    assert "No series with out-of-range values found." in captured.out
    assert "'series_1': 'exog1' has values outside" in captured.out
    assert "'series_1': 'exog2' has values outside" in captured.out


def test_summary_both_out_of_range(capsys):
    """
    Test _summary with both out-of-range series and exogenous variables.
    """
    out_of_range_series = ["series1"]
    out_of_range_series_ranges = [(1.0, 10.0)]
    out_of_range_exog = ["exog1"]
    out_of_range_exog_ranges = [(0.0, 5.0)]
    out_of_range_exog_series_id = [None]

    RangeDriftDetector._summary(
        out_of_range_series=out_of_range_series,
        out_of_range_series_ranges=out_of_range_series_ranges,
        out_of_range_exog=out_of_range_exog,
        out_of_range_exog_ranges=out_of_range_exog_ranges,
        out_of_range_exog_series_id=out_of_range_exog_series_id
    )

    captured = capsys.readouterr()
    assert "'series1' has values outside the observed range [1.00000, 10.00000]." in captured.out
    assert "'exog1' has values outside the observed range [0.00000, 5.00000]." in captured.out
