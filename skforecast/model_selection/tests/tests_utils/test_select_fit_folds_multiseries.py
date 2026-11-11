# Unit test _select_fit_folds_multiseries
# ==============================================================================
import pytest
from skforecast.model_selection._utils import _select_fit_folds_multiseries


def make_folds(refit_flags):
    """
    Create folds with the given refit flags (`fold[5]`), the indexes are not
    used by the function.
    """
    return [
        [i, [0, 10], [7, 10], [10, 15], [10, 15], refit]
        for i, refit in enumerate(refit_flags)
    ]


@pytest.mark.parametrize(
    'refit_flags, levels_predict_folds, expected',
    [
        # Refit in every fold: only the folds with levels to predict are fitted
        ([True, True, True, True], [['l1'], [], [], ['l2']],
         [True, False, False, True]),
        # Intermittent refit: a refit fold without levels to predict is fitted
        # if a following fold without refit predicts with its model
        ([True, False, True, False], [['l1'], [], [], ['l2']],
         [True, False, True, False]),
        # Intermittent refit: no fold of the second block predicts
        ([True, False, True, False], [['l1'], ['l1'], [], []],
         [True, False, False, False]),
        # No refit: the initial fit is only needed if a fold predicts
        ([True, False, False], [[], [], ['l1']], [True, False, False]),
        ([True, False, False], [[], [], []], [False, False, False]),
        # Externally fitted forecaster: no fold is fitted
        ([False, False], [['l1'], ['l1']], [False, False]),
    ],
    ids=lambda dt: f'{dt}'
)
def test_select_fit_folds_multiseries_output(
    refit_flags, levels_predict_folds, expected
):
    """
    Test the forecaster is only fitted in a refit fold whose model predicts at
    least one level, in that fold or in the following folds without refit.
    """
    folds = make_folds(refit_flags)

    fit_folds = _select_fit_folds_multiseries(
                    folds                = folds,
                    levels_predict_folds = levels_predict_folds
                )

    assert fit_folds == expected
