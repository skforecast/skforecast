"""
Smoke test of an installed skforecast distribution. Run by
`.github/workflows/publish-pypi.yml` on the wheel just built and on the version
installed from the index.

It must run outside the repository (so that the `skforecast/` folder does not
shadow the installed package) and with `-I`, in an environment that only has
the core dependencies:

    cd "$RUNNER_TEMP"
    venv/bin/python -I path/to/smoke_test.py <expected version>
"""

from __future__ import annotations

import importlib.util
import sys
import sysconfig
from importlib.metadata import version as installed_version
from pathlib import Path

# Optional dependencies that must not be installed: the test checks that
# skforecast works with the core dependencies alone.
OPTIONAL = ["matplotlib", "statsmodels", "keras", "torch", "lightgbm"]


def main(expected: str) -> None:
    import skforecast

    location = Path(skforecast.__file__).resolve()
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    if site_packages not in location.parents:
        raise SystemExit(
            f"skforecast is imported from {location}, not from {site_packages}"
        )
    if skforecast.__version__ != expected:
        raise SystemExit(f"skforecast.__version__ is {skforecast.__version__}")
    if installed_version("skforecast") != expected:
        raise SystemExit(f"the metadata says {installed_version('skforecast')}")

    installed = [name for name in OPTIONAL if importlib.util.find_spec(name)]
    if installed:
        raise SystemExit(f"optional dependencies are installed: {installed}")

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from skforecast.recursive import ForecasterRecursive

    # A linear trend: the recursive forecast of a linear model with lags
    # continues it exactly.
    y = pd.Series(
        10 + 0.5 * np.arange(60, dtype=float),
        index=pd.date_range("2020-01-01", periods=60, freq="D"),
        name="y",
    )
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)

    expected_index = pd.date_range(y.index[-1], periods=6, freq="D")[1:]
    if not predictions.index.equals(expected_index):
        raise SystemExit(f"unexpected index of the predictions: {predictions.index}")
    np.testing.assert_allclose(
        predictions.to_numpy(), 10 + 0.5 * np.arange(60, 65), rtol=1e-8
    )

    print(
        f"skforecast {skforecast.__version__} from {location.parent} "
        f"(Python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"pandas {pd.__version__}): forecast OK"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
