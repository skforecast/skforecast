# Unit test _get_source_with_imports
# ==============================================================================
import inspect
import pytest
import numpy as np
import pandas as pd
from numpy import where as np_where
from ...utils import _get_source_with_imports

MIN_WEIGHT = 0.5
month = 3


def weights_modules(index):  # pragma: no cover
    """
    Use modules, also inside a comprehension, and an attribute with the name
    of a global variable (`month`).
    """
    weights = [np.float64(date.month > 1) for date in index]
    return pd.Series(weights, index=index).to_numpy()


def weights_imported_function(index):  # pragma: no cover
    """
    Use a function imported with an alias.
    """
    return np_where(index >= '2004-01-01', 1, 0)


def weights_signature(
    index: pd.DatetimeIndex, weight: np.float64 = np.float64(1.0)
) -> list:  # pragma: no cover
    """
    Use modules only in the signature (annotations and default values), which
    is evaluated when the module is imported.
    """
    return [weight] * len(index)


def weights_helper(index):  # pragma: no cover
    """
    Function used by `weights_not_importable`.
    """
    return np.ones(len(index))


def weights_not_importable(index):  # pragma: no cover
    """
    Use a global constant and a function defined in the '__main__' namespace.
    """
    return np.maximum(weights_helper(index), MIN_WEIGHT)


def make_weights_closure(value):  # pragma: no cover
    """
    Create a weight function that uses a variable of the enclosing function.
    """
    def weights_closure(index):
        return np.full(len(index), value)

    return weights_closure


@pytest.mark.parametrize(
    "fun, expected_imports",
    [
        (weights_modules, "import numpy as np\nimport pandas as pd"),
        (weights_imported_function, "from numpy import where as np_where"),
        (weights_signature, "import numpy as np\nimport pandas as pd"),
    ],
    ids=['modules', 'imported_function', 'signature']
)
def test_get_source_with_imports_output(fun, expected_imports):
    """
    Test that _get_source_with_imports writes the import statements of the
    modules (also those used inside a comprehension or only in the signature)
    and of the functions imported with an alias before the source code of the
    function, and that an attribute with the name of a global variable is not
    taken as a global.
    """
    source_code, names_not_imported = _get_source_with_imports(fun)

    assert source_code == expected_imports + "\n\n\n" + inspect.getsource(fun)
    assert names_not_imported == []


@pytest.mark.parametrize(
    "fun, expected_names",
    [
        (weights_not_importable, ['MIN_WEIGHT', 'weights_helper']),
        (make_weights_closure(2), ['value']),
    ],
    ids=['global_variables', 'closure']
)
def test_get_source_with_imports_names_not_imported(
    fun, expected_names, monkeypatch
):
    """
    Test that _get_source_with_imports returns the names of the objects that
    cannot be written as an import: a global constant, a function defined in
    the '__main__' namespace and a variable of an enclosing function.
    """
    monkeypatch.setattr(weights_helper, '__module__', '__main__')

    source_code, names_not_imported = _get_source_with_imports(fun)

    assert source_code == "import numpy as np\n\n\n" + inspect.getsource(fun)
    assert names_not_imported == expected_names
