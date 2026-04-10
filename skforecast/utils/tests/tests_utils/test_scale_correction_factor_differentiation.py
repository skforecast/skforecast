# Unit test scale_correction_factor_differentiation
# ==============================================================================
import numpy as np
import pytest
from skforecast.utils import scale_correction_factor_differentiation


@pytest.mark.parametrize(
    'correction_factor, steps, differentiation_order, expected',
    [
        # d=1, scalar cf=1.0, 5 steps: scaling = sqrt([1, 2, 3, 4, 5])
        (
            1.0,
            5,
            1,
            np.array([1.0, 1.4142135623730951, 1.7320508075688772,
                       2.0, 2.23606797749979]),
        ),
        # d=1, scalar cf=2.5, 4 steps
        (
            2.5,
            4,
            1,
            np.array([2.5, 3.5355339059327378, 4.330127018922193, 5.0]),
        ),
        # d=2, scalar cf=1.0, 5 steps: psi_j = j, scaling = sqrt(cumsum(j^2))
        (
            1.0,
            5,
            2,
            np.array([1.0, 2.23606797749979, 3.7416573867739413,
                       5.477225575051661, 7.416198487095663]),
        ),
        # d=1, scalar cf=3.0, single step
        (
            3.0,
            1,
            1,
            np.array([3.0]),
        ),
        # d=3, scalar cf=1.0, 6 steps: psi_j = comb(j+1, 2)
        (
            1.0,
            6,
            3,
            np.array([1.0, 3.1622776601683795, 6.782329983125268,
                       12.083045973594572, 19.261360284258224,
                       28.495613697550013]),
        ),
    ],
    ids=[
        'd1_scalar_1',
        'd1_scalar_2.5',
        'd2_scalar_1',
        'd1_single_step',
        'd3_scalar_1',
    ]
)
def test_scale_correction_factor_differentiation_output_when_scalar(
    correction_factor, steps, differentiation_order, expected
):
    """
    Test output of scale_correction_factor_differentiation when
    correction_factor is a scalar (float) for differentiation orders
    d=1, d=2 and d=3.
    """
    result = scale_correction_factor_differentiation(
        correction_factor     = correction_factor,
        steps                 = steps,
        differentiation_order = differentiation_order,
    )
    assert isinstance(result, np.ndarray)
    assert len(result) == steps
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    'correction_factor, steps, differentiation_order, expected',
    [
        # d=1, array cf, 3 steps
        (
            np.array([1.0, 2.0, 3.0]),
            3,
            1,
            np.array([1.0, 2.8284271247461903, 5.196152422706632]),
        ),
        # d=2, array cf, 4 steps
        (
            np.array([0.5, 1.5, 2.5, 3.5]),
            4,
            2,
            np.array([0.5, 3.3541019662496847, 9.354143466934854,
                       19.170289512680814]),
        ),
    ],
    ids=[
        'd1_array',
        'd2_array',
    ]
)
def test_scale_correction_factor_differentiation_output_when_array(
    correction_factor, steps, differentiation_order, expected
):
    """
    Test output of scale_correction_factor_differentiation when
    correction_factor is a 1D numpy array (binned residuals).
    """
    result = scale_correction_factor_differentiation(
        correction_factor     = correction_factor,
        steps                 = steps,
        differentiation_order = differentiation_order,
    )
    assert isinstance(result, np.ndarray)
    assert len(result) == steps
    np.testing.assert_array_almost_equal(result, expected)


def test_scale_correction_factor_differentiation_output_is_monotonically_increasing():
    """
    Test that the scaled correction factor is monotonically increasing
    when correction_factor is a positive scalar (intervals widen over
    the forecast horizon).
    """
    result = scale_correction_factor_differentiation(
        correction_factor     = 2.0,
        steps                 = 20,
        differentiation_order = 1,
    )
    expected = np.array([
        2.0, 2.8284271247461903, 3.4641016151377544, 4.0,
        4.47213595499958, 4.898979485566356, 5.291502622129181,
        5.656854249492381, 6.0, 6.324555320336759,
        6.6332495807108, 6.928203230275509, 7.211102550927978,
        7.483314773547883, 7.745966692414834, 8.0,
        8.246211251235321, 8.48528137423857, 8.717797887081348,
        8.94427190999916,
    ])
    np.testing.assert_array_almost_equal(result, expected)
    assert np.all(np.diff(result) > 0)
