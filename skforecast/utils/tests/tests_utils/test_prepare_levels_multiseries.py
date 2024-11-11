# Unit test prepare_levels_multiseries
# ==============================================================================
import pytest
from skforecast.utils import prepare_levels_multiseries


@pytest.mark.parametrize("levels, expected_levels, expected_type", 
                         [(None, ['l1', 'l2', 'l3'], False),
                          ('l1', ['l1'], False),
                          (['l1'], ['l1'], True)], 
                         ids=lambda levels: f'levels: {levels}')
def test_output_prepare_levels_multiseries(levels, expected_levels, expected_type):
    """
    Test output prepare_levels_multiseries for different inputs.
    """
    X_train_series_names_in_ = ['l1', 'l2', 'l3']
    levels, input_levels_is_list = prepare_levels_multiseries(
        X_train_series_names_in_=X_train_series_names_in_, levels=levels
    )
    
    assert levels == expected_levels
    assert input_levels_is_list is expected_type
