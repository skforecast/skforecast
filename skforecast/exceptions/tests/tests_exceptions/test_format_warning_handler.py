# Unit test format_warning_handler and rich_warning_handler
# ==============================================================================
import pytest
import warnings
from skforecast.exceptions.exceptions import (
    format_warning_handler,
    rich_warning_handler,
    set_warnings_style,
    LongTrainingWarning,
    MissingValuesWarning,
    warn_skforecast_categories
)


def test_format_warning_handler_skforecast_warnings(capsys):
    """
    Test format_warning_handler prints formatted box for skforecast warnings
    and falls back to default for non-skforecast warnings.
    """
    # Test with skforecast warning - should print formatted box
    message = LongTrainingWarning("Test warning message")
    format_warning_handler(
        message=message,
        category=LongTrainingWarning,
        filename="test_file.py",
        lineno=42
    )
    
    captured = capsys.readouterr()
    assert "LongTrainingWarning" in captured.out
    assert "Test warning message" in captured.out
    assert "╭" in captured.out  # Box borders
    assert "╰" in captured.out
    
    # Test fallback for non-skforecast warning
    fallback_called = []
    warnings._original_showwarning = lambda *args: fallback_called.append(True)
    
    format_warning_handler(
        message="Standard warning",
        category=UserWarning,
        filename="test.py",
        lineno=1
    )
    assert len(fallback_called) == 1


def test_rich_warning_handler_skforecast_warnings(capsys):
    """
    Test rich_warning_handler prints panel with location, suppress info,
    and falls back for non-skforecast warnings.
    """
    message = MissingValuesWarning("Missing data detected")
    rich_warning_handler(
        message=message,
        category=MissingValuesWarning,
        filename="my_script.py",
        lineno=123
    )
    
    captured = capsys.readouterr()
    assert "MissingValuesWarning" in captured.out
    assert "Missing data detected" in captured.out
    assert "my_script.py:123" in captured.out
    assert "Suppress" in captured.out
    assert "warnings.simplefilter" in captured.out
    
    # Test fallback
    fallback_called = []
    warnings._original_showwarning = lambda *args: fallback_called.append(True)
    
    rich_warning_handler(
        message="Standard warning",
        category=UserWarning,
        filename="test.py",
        lineno=1
    )
    assert len(fallback_called) == 1


def test_set_warnings_style():
    """
    Test set_warnings_style switches between 'skforecast' and 'default' handlers.
    """
    set_warnings_style(style='skforecast')
    assert warnings.showwarning == rich_warning_handler
    assert hasattr(warnings, '_original_showwarning')
    
    original = warnings._original_showwarning
    set_warnings_style(style='default')
    assert warnings.showwarning == original


@pytest.mark.parametrize("warning_class", warn_skforecast_categories)
def test_handlers_accept_all_skforecast_warning_categories(warning_class, capsys):
    """
    Test that both handlers work with all 14 skforecast warning types.
    """
    message = warning_class("Test message")
    
    format_warning_handler(
        message=message, category=warning_class, filename="test.py", lineno=1
    )
    captured = capsys.readouterr()
    assert warning_class.__name__ in captured.out
    
    rich_warning_handler(
        message=message, category=warning_class, filename="test.py", lineno=1
    )
    captured = capsys.readouterr()
    assert warning_class.__name__ in captured.out
