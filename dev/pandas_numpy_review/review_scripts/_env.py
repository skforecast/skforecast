"""
Shared header of the review scripts. Import it first: `import _env`.

- NO_PYARROW=1 blocks the import of pyarrow (pandas 3 without pyarrow).
- `_env.versions()` returns the versions to print next to every number.
"""
import os
import sys

if os.environ.get("NO_PYARROW") == "1":
    class _BlockPyarrow:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")

    sys.meta_path.insert(0, _BlockPyarrow)


def versions() -> str:
    import numpy
    import pandas
    import sklearn
    import skforecast

    try:
        import pyarrow  # noqa: F401
        pa = pyarrow.__version__
    except ModuleNotFoundError:
        pa = "no"
    extra = []
    for name in ("lightgbm", "xgboost", "catboost"):
        if name in sys.modules:
            extra.append(f"{name} {sys.modules[name].__version__}")
    return (
        f"[python {sys.version.split()[0]}, pandas {pandas.__version__}, pyarrow {pa}, "
        f"numpy {numpy.__version__}, scikit-learn {sklearn.__version__}"
        + (", " + ", ".join(extra) if extra else "")
        + f", skforecast {skforecast.__version__} @ {os.path.dirname(skforecast.__file__)}]"
    )
