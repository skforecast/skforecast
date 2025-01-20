# Unit test create_mean_pinball_loss
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.metrics import create_mean_pinball_loss
from sklearn.metrics import mean_pinball_loss


def test_create_mean_pinball_loss_valid_input_alpha():
    """
    Check valid input parameter alpha.
    """
    create_mean_pinball_loss(alpha=0.5)
    create_mean_pinball_loss(alpha=0)
    create_mean_pinball_loss(alpha=1)

    msg = "alpha must be between 0 and 1, both inclusive."
    with pytest.raises(ValueError, match=msg):
        create_mean_pinball_loss(alpha=-1)

    with pytest.raises(ValueError, match=msg):
        create_mean_pinball_loss(alpha=2)


def test_create_pinball_loss_output_equivalent_to_sklearn():
    """
    Check that the output of the function is equivalent to the output of the function
    `mean_pinball_loss` from scikit-learn.
    """
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([2, 6, 4, 3, 1])

    for alpha in [0.1, 0.5, 0.9]:
        pinball_loss = create_mean_pinball_loss(alpha=alpha)
        assert pinball_loss(y_true, y_pred) == mean_pinball_loss(y_true, y_pred, alpha=alpha)
