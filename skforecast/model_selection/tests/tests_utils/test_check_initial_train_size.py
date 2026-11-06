# Unit test _check_initial_train_size
# ==============================================================================
import re
import pytest
import pandas as pd
from skforecast.model_selection._utils import _check_initial_train_size


@pytest.mark.parametrize(
    "window_size, allow_window_size, bound",
    [(95, False, "greater than"), (96, True, "at least")],
    ids=lambda dt: f"window_size, allow_window_size, bound: {dt}"
)
def test_check_initial_train_size_ValueError_when_window_size_leaves_no_valid_value(
    window_size, allow_window_size, bound
):
    """
    Test ValueError is raised when no `initial_train_size` is valid because
    the `window_size` is too large for the data, whatever the value received.
    """
    err_msg = re.escape(
        f"The `window_size` of the forecaster ({window_size}) leaves no valid "
        f"`initial_train_size`, which must be {bound} the `window_size` and "
        f"smaller than the length of `y` (96). Reduce the `window_size` (lags, "
        f"window features, differentiation...) or use more data."
    )
    with pytest.raises(ValueError, match=err_msg):
        _check_initial_train_size(
            initial_train_size       = 50,
            initial_train_size_input = 50,
            data_length              = 96,
            window_size              = window_size,
            allow_window_size        = allow_window_size
        )


@pytest.mark.parametrize(
    "initial_train_size_input",
    ["2020-01-03", pd.Timestamp("2020-01-03")],
    ids=lambda dt: f"initial_train_size_input: {dt!r}"
)
def test_check_initial_train_size_ValueError_shows_date_received(
    initial_train_size_input
):
    """
    Test the ValueError shows the date received and the number of observations
    of the training set when `initial_train_size` is a date.
    """
    err_msg = re.escape(
        "`initial_train_size` must be greater than the `window_size` of the "
        "forecaster (3), so that the first training set has data to train with. "
        "Got '2020-01-03"
    )
    with pytest.raises(ValueError, match=err_msg) as exc_info:
        _check_initial_train_size(
            initial_train_size       = 3,
            initial_train_size_input = initial_train_size_input,
            data_length              = 96,
            window_size              = 3
        )

    assert str(exc_info.value).endswith("3 observations in the training set.")


def test_check_initial_train_size_ForecasterStats_initial_train_size_equal_to_window_size():
    """
    Test that `initial_train_size` equal to the `window_size` is allowed with
    `allow_window_size=True` (ForecasterStats, whose `window_size` is ignored),
    and that a smaller value raises a ValueError.
    """
    _check_initial_train_size(
        initial_train_size       = 1,
        initial_train_size_input = 1,
        data_length              = 96,
        window_size              = 1,
        allow_window_size        = True
    )

    err_msg = re.escape(
        "`initial_train_size` must be greater than or equal to the `window_size` "
        "of the forecaster (1), so that the first training set has data to train "
        "with. Got 0."
    )
    with pytest.raises(ValueError, match=err_msg):
        _check_initial_train_size(
            initial_train_size       = 0,
            initial_train_size_input = 0,
            data_length              = 96,
            window_size              = 1,
            allow_window_size        = True
        )


def test_check_initial_train_size_window_size_None():
    """
    Test that, without `window_size` (ForecasterFoundation, whose `window_size`
    is the maximum context of the model), `initial_train_size` only needs to be
    greater than 0, even if the data is shorter than the context.
    """
    _check_initial_train_size(
        initial_train_size       = 1,
        initial_train_size_input = 1,
        data_length              = 50,
        data_length_name         = "the length of the span of `series`",
        window_size              = None
    )

    err_msg = re.escape(
        "`initial_train_size` must be greater than 0, so that the first training "
        "set has data to train with. Got 0."
    )
    with pytest.raises(ValueError, match=err_msg):
        _check_initial_train_size(
            initial_train_size       = 0,
            initial_train_size_input = 0,
            data_length              = 50,
            window_size              = None
        )
