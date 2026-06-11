# Unit test stats __init__
# ==============================================================================
import re
import pytest
import skforecast.stats as stats


def test_stats_getattr_ImportError_when_statsmodels_is_missing(monkeypatch):
    """
    Test that lazy stats imports raise the skforecast optional dependency error
    when statsmodels is missing.
    """
    err_msg = "\n'statsmodels' is an optional dependency not included"

    def import_module_mock(module_name, package=None):
        raise ModuleNotFoundError(
            "No module named 'statsmodels'", name="statsmodels"
        )

    def check_optional_dependency_mock(package_name):
        assert package_name == "statsmodels"
        raise ImportError(err_msg)

    monkeypatch.setattr(stats, "import_module", import_module_mock)
    monkeypatch.setattr(
        stats, "check_optional_dependency", check_optional_dependency_mock
    )

    with pytest.raises(ImportError, match=re.escape(err_msg)):
        stats.__getattr__("Sarimax")


def test_stats_getattr_ModuleNotFoundError_when_module_is_not_statsmodels(monkeypatch):
    """
    Test that lazy stats imports do not hide unrelated missing module errors.
    """
    err_msg = "No module named 'unknown_package'"

    def import_module_mock(module_name, package=None):
        raise ModuleNotFoundError(err_msg, name="unknown_package")

    monkeypatch.setattr(stats, "import_module", import_module_mock)

    with pytest.raises(ModuleNotFoundError, match=re.escape(err_msg)):
        stats.__getattr__("Sarimax")