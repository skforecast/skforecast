"""
Shared helpers for the repro scripts of `dev/review_028/`.

Files that must travel between environments (saved with pandas 2, loaded with
pandas 3) are written to `$REVIEW028_OUT`, by default a folder in the system
temporary directory, never inside the repository.
"""
import os
import tempfile
from pathlib import Path

OUT = Path(
    os.environ.get(
        "REVIEW028_OUT", Path(tempfile.gettempdir()) / "skforecast_review_028"
    )
)
OUT.mkdir(parents=True, exist_ok=True)


def versions() -> str:
    import numpy
    import pandas
    import skforecast

    return (
        f"skforecast {skforecast.__version__}, pandas {pandas.__version__}, "
        f"numpy {numpy.__version__}"
    )


def attempt(label: str, fn) -> None:
    """Run `fn` and print its result or the exception it raises."""
    try:
        out = fn()
        print(f"{label}: OK" + ("" if out is None else f" {out}"))
    except Exception as exc:
        notes = getattr(exc, "__notes__", None)
        print(f"{label}: {type(exc).__name__}: {str(exc)[:200]}")
        if notes:
            print(f"    notes: {notes}")
