# Unit test show_versions
# ==============================================================================
from skforecast.utils import show_versions


def test_show_versions():
    """
    Test show versions function.
    """
    
    show_versions()
    s = show_versions(as_str=True)

    assert isinstance(s, str)
    assert "System" in s
    assert "Python dependencies" in s


def test_show_versions_output_lists_core_and_optional_dependencies():
    """
    Test that the output has one line per core and optional dependency,
    including those that are not installed.
    """

    vers_info = show_versions(as_str=True)
    listed_names = [
        line.split(":")[0].strip() for line in vers_info.splitlines()
        if ":" in line
    ]

    expected_names = [
        "skforecast", "pip", "setuptools", "numpy", "pandas", "tqdm",
        "scikit-learn", "scipy", "optuna", "joblib", "numba", "rich",
        "statsmodels", "matplotlib", "keras", "torch", "lightgbm", "xgboost",
        "catboost", "skops", "cloudpickle",
    ]
    for name in expected_names:
        assert name in listed_names
