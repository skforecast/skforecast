# Unit test runtime_deprecated decorator
# ==============================================================================
import pytest
from skforecast.exceptions import runtime_deprecated


def test_runtime_deprecated_function():
    """
    Test runtime_deprecated decorator on functions: emits warning with correct
    message, preserves return value, args/kwargs, and sets metadata.
    """
    @runtime_deprecated(
        replacement="new_func()", 
        version="0.19.0", 
        removal="1.0.0"
    )
    def deprecated_func(x, y, c=10):
        return x + y + c
    
    # Check warning is emitted with correct content
    with pytest.warns(FutureWarning) as record:
        result = deprecated_func(2, 3, c=5)
    
    warning_message = str(record[0].message)
    assert "deprecated_func() is deprecated" in warning_message
    assert "since version 0.19.0" in warning_message
    assert "use new_func() instead" in warning_message
    assert "will be removed in version 1.0.0" in warning_message
    
    # Check return value and function behavior preserved
    assert result == 10
    
    # Check metadata attributes
    assert deprecated_func.__deprecated__ is True
    assert deprecated_func.__replacement__ == "new_func()"
    assert deprecated_func.__version__ == "0.19.0"
    assert deprecated_func.__removal__ == "1.0.0"
    assert deprecated_func.__name__ == "deprecated_func"


def test_runtime_deprecated_class():
    """
    Test runtime_deprecated decorator on classes: emits warning, preserves
    __init__ behavior, and sets metadata.
    """
    @runtime_deprecated(
        replacement="NewClass", 
        version="0.15.0", 
        removal="1.0.0"
    )
    class DeprecatedClass:
        def __init__(self, name, value=100):
            self.name = name
            self.value = value
    
    # Check warning is emitted with correct content
    with pytest.warns(FutureWarning) as record:
        obj = DeprecatedClass("test", value=200)
    
    warning_message = str(record[0].message)
    assert "DeprecatedClass class is deprecated" in warning_message
    assert "since version 0.15.0" in warning_message
    assert "use NewClass instead" in warning_message
    assert "will be removed in version 1.0.0" in warning_message
    
    # Check class behavior preserved
    assert obj.name == "test"
    assert obj.value == 200
    
    # Check metadata attributes
    assert DeprecatedClass.__deprecated__ is True
    assert DeprecatedClass.__replacement__ == "NewClass"


def test_runtime_deprecated_minimal_and_custom_category():
    """
    Test decorator with no optional params and with custom warning category.
    """
    # Minimal usage
    @runtime_deprecated()
    def minimal_func():
        return "minimal"
    
    with pytest.warns(FutureWarning, match="minimal_func\\(\\) is deprecated"):
        result = minimal_func()
    
    assert result == "minimal"
    assert minimal_func.__replacement__ is None
    assert minimal_func.__version__ is None
    
    # Custom category
    @runtime_deprecated(category=DeprecationWarning)
    def custom_category_func():
        pass
    
    with pytest.warns(DeprecationWarning, match="custom_category_func\\(\\) is deprecated"):
        custom_category_func()


def test_runtime_deprecated_raises_error_for_non_callable():
    """
    Test that decorator raises TypeError for non-function/class objects.
    """
    decorator = runtime_deprecated()
    
    with pytest.raises(TypeError, match="can only be used on functions or classes"):
        decorator("not a function")
    
    with pytest.raises(TypeError, match="can only be used on functions or classes"):
        decorator(42)
