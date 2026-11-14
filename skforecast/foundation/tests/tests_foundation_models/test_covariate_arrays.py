# Unit test conversion of covariate columns to numpy arrays in the adapters
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import (
    ChronosAdapter,
    NoriAdapter,
    T0Adapter,
    TimesFM3Adapter,
    TimestralAdapter,
    TiRex2Adapter,
    TSICLAdapter,
)


@pytest.mark.parametrize(
    "converter, expected_dtype",
    [
        (ChronosAdapter._to_covariate_array, np.float32),
        (TimesFM3Adapter._to_covariate_array, np.float32),
        (TSICLAdapter._to_covariate_array, np.float32),
        (TiRex2Adapter._to_covariate_array, np.float32),
        (T0Adapter._to_float_array, np.float32),
        (NoriAdapter._to_float_array, np.float32),
        (TimestralAdapter._to_float_array, np.float64),
    ],
    ids=["chronos", "timesfm3", "tsicl", "tirex2", "t0", "nori", "timestral"],
)
@pytest.mark.parametrize(
    "dtype, values",
    [
        ("Int64", [1, 2, None, 4]),
        ("Float64", [1.5, 2.5, None, 4.5]),
        ("boolean", [True, False, None, True]),
    ],
    ids=["Int64", "Float64", "boolean"],
)
def test_covariate_arrays_nullable_dtype_with_missing_values_to_nan(
    converter, expected_dtype, dtype, values
):
    """
    Test that the adapters convert a covariate with a nullable pandas dtype
    and missing values to a float array with NaN in the missing positions.
    """
    col_data = pd.Series(values, dtype=dtype, name="feat")
    result = converter(col_data)

    expected = np.array(
        [np.nan if v is None else float(v) for v in values], dtype=expected_dtype
    )

    assert result.dtype == expected_dtype
    np.testing.assert_array_equal(result, expected)
