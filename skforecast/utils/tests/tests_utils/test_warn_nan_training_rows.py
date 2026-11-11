# Unit test _warn_nan_training_rows
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
from skforecast.exceptions import MissingValuesWarning
from skforecast.utils import _warn_nan_training_rows

FIRST_SENTENCE = "Some rows of the training matrix have missing values."
PREDICTORS = "with NaN predictors (lags, window features or exog)."
ACTION_TRUE = (
    "To keep the rows with NaN predictors, set "
    "`forecaster.dropna_from_series = False` (the estimator must support NaN)."
)
ACTION_FALSE = (
    "The estimator must support NaN. To drop those rows, set "
    "`forecaster.dropna_from_series = True`."
)


def warning_message(**kwargs) -> str:
    """
    Call `_warn_nan_training_rows` and return the text of the single warning
    raised.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _warn_nan_training_rows(**kwargs)
    assert len(record) == 1
    assert issubclass(record[0].category, MissingValuesWarning)

    return record[0].message.message


@pytest.mark.parametrize(
    "n_nan_target, n_nan_predictors, dropna_from_series, counts, action",
    [
        (2, 10, True,
         f"12 of 95 rows are dropped: 2 with a NaN target and 10 {PREDICTORS}",
         ACTION_TRUE),
        (1, 1, True,
         f"2 of 95 rows are dropped: 1 with a NaN target and 1 {PREDICTORS}",
         ACTION_TRUE),
        (0, 1, True, f"1 of 95 rows is dropped, {PREDICTORS}", ACTION_TRUE),
        (0, 3, True, f"3 of 95 rows are dropped, {PREDICTORS}", ACTION_TRUE),
        (2, 0, True, "2 of 95 rows are dropped, with a NaN target.", ""),
        (1, 0, False, "1 of 95 rows is dropped, with a NaN target.", ""),
        (2, 10, False,
         f"2 of 95 rows are dropped (NaN target) and 10 are kept {PREDICTORS}",
         ACTION_FALSE),
        (1, 1, False,
         f"1 of 95 rows is dropped (NaN target) and 1 is kept {PREDICTORS}",
         ACTION_FALSE),
        (0, 1, False, f"1 of 95 rows is kept {PREDICTORS}", ACTION_FALSE),
        (0, 4, False, f"4 of 95 rows are kept {PREDICTORS}", ACTION_FALSE),
    ],
    ids=lambda x: f"{x}"[:12]
)
def test_warn_nan_training_rows_message_of_each_variant(
    n_nan_target, n_nan_predictors, dropna_from_series, counts, action
):
    """
    Test the text of each variant of the warning (both counts, only NaN
    predictors, only NaN target, with `dropna_from_series` True and False),
    in singular and plural. The zero counts are not shown, and the action is
    only given when there are rows with NaN predictors.
    """
    message = warning_message(
                  n_nan_target       = n_nan_target,
                  n_nan_predictors   = n_nan_predictors,
                  n_rows             = 95,
                  dropna_from_series = dropna_from_series,
                  series_name        = "`y`"
              )

    expected = (
        f"{FIRST_SENTENCE} {counts} They come from interspersed NaNs in `y` "
        f"or `exog`."
    )
    if action:
        expected = f"{expected} {action}"

    assert message == expected


@pytest.mark.parametrize(
    "n_nan_target, n_nan_predictors, dropna_from_series, counts",
    [
        ([10, 10, 10], [24, 30, 33], True,
         f"34 to 43 of 453 rows are dropped per step: 10 with a NaN target and "
         f"24 to 33 {PREDICTORS}"),
        # The total of each step is the sum of its two counts, not the sum of
        # the ranges (which would be 24 to 27).
        ([0, 2, 1], [26, 24, 25], True,
         f"26 of 453 rows are dropped per step: 0 to 2 with a NaN target and "
         f"24 to 26 {PREDICTORS}"),
        ([2, 2], [48, 48], True,
         f"50 of 453 rows are dropped per step: 2 with a NaN target and "
         f"48 {PREDICTORS}"),
        ([1, 1], [0, 0], True,
         "1 of 453 rows is dropped per step, with a NaN target."),
        ([0, 1], [0, 0], False,
         "0 to 1 of 453 rows are dropped per step, with a NaN target."),
        ([0, 1], [24, 24], False,
         f"0 to 1 of 453 rows are dropped per step (NaN target) and 24 are "
         f"kept {PREDICTORS}"),
        ([0, 0], [1, 1], False, f"1 of 453 rows is kept per step {PREDICTORS}"),
        ([0, 0], [1, 2], True, f"1 to 2 of 453 rows are dropped per step, {PREDICTORS}"),
        (np.array([3]), np.array([5]), True,
         f"8 of 453 rows are dropped per step: 3 with a NaN target and "
         f"5 {PREDICTORS}"),
    ],
    ids=lambda x: f"{x}"[:12]
)
def test_warn_nan_training_rows_message_per_step(
    n_nan_target, n_nan_predictors, dropna_from_series, counts
):
    """
    Test the text of the warning of the direct forecasters, with the counts of
    each step: a single number if all the steps have the same, a range
    ("a to b", plural) otherwise.
    """
    message = warning_message(
                  n_nan_target       = n_nan_target,
                  n_nan_predictors   = n_nan_predictors,
                  n_rows             = 453,
                  dropna_from_series = dropna_from_series,
                  series_name        = "`series`"
              )

    assert message.startswith(f"{FIRST_SENTENCE} {counts} They come from ")
    assert "interspersed NaNs in `series` or `exog`." in message
    if np.max(n_nan_predictors) > 0:
        action = ACTION_TRUE if dropna_from_series else ACTION_FALSE
        assert message.endswith(action)
    else:
        assert message.endswith("`exog`.")


@pytest.mark.parametrize(
    "n_nan_target, n_nan_predictors",
    [(0, 0), ([0, 0], [0, 0])],
    ids=lambda x: f"{x}"
)
def test_warn_nan_training_rows_no_warning_when_no_rows_with_NaN(
    n_nan_target, n_nan_predictors
):
    """
    Test no warning is raised when no row has a NaN target or NaN predictors.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _warn_nan_training_rows(
            n_nan_target       = n_nan_target,
            n_nan_predictors   = n_nan_predictors,
            n_rows             = 95,
            dropna_from_series = True,
            series_name        = "`y`"
        )

    assert len(record) == 0


def test_warn_nan_training_rows_first_sentence_does_not_change():
    """
    Test the first sentence of the warning, up to the first period followed by
    a space, is the same and has no digits in every variant, so the searches
    group the warnings of their candidates in one.
    """
    first_sentences = set()
    for n_nan_target in (0, 1, 2, [0, 1], [2, 2]):
        for n_nan_predictors in (0, 1, 10, [3, 5], [1, 1]):
            if np.max(n_nan_target) == 0 and np.max(n_nan_predictors) == 0:
                continue
            if np.ndim(n_nan_target) != np.ndim(n_nan_predictors):
                continue
            for dropna_from_series in (True, False):
                for series_name in ("`y`", "`series`"):
                    message = warning_message(
                                  n_nan_target       = n_nan_target,
                                  n_nan_predictors   = n_nan_predictors,
                                  n_rows             = 95,
                                  dropna_from_series = dropna_from_series,
                                  series_name        = series_name
                              )
                    first_sentences.add(re.split(r"\.(?=\s)", message)[0])

    assert first_sentences == {FIRST_SENTENCE[:-1]}
